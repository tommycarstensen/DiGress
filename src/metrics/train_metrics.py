import torch
from torch import Tensor
import torch.nn as nn
from torchmetrics import Metric, MeanSquaredError, MetricCollection
import time
import wandb
from src.metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchMSE, SumExceptBatchKL, CrossEntropyMetric, \
    ProbabilityMetric, NLL
import torch.nn.functional as F  # tilfoejet pga calculate_valence_penalty

class NodeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class EdgeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class TrainLoss(nn.Module):
    def __init__(self):
        super(TrainLoss, self).__init__()
        self.train_node_mse = NodeMSE()
        self.train_edge_mse = EdgeMSE()
        self.train_y_mse = MeanSquaredError()

    def forward(self, masked_pred_epsX, masked_pred_epsE, pred_y, true_epsX, true_epsE, true_y, log: bool):
        mse_X = self.train_node_mse(masked_pred_epsX, true_epsX) if true_epsX.numel() > 0 else 0.0
        mse_E = self.train_edge_mse(masked_pred_epsE, true_epsE) if true_epsE.numel() > 0 else 0.0
        mse_y = self.train_y_mse(pred_y, true_y) if true_y.numel() > 0 else 0.0
        mse = mse_X + mse_E + mse_y

        if log:
            to_log = {'train_loss/batch_mse': mse.detach(),
                      'train_loss/node_MSE': self.train_node_mse.compute(),
                      'train_loss/edge_MSE': self.train_edge_mse.compute(),
                      'train_loss/y_mse': self.train_y_mse.compute()}
            if wandb.run:
                wandb.log(to_log, commit=True)

        return mse

    def reset(self):
        for metric in (self.train_node_mse, self.train_edge_mse, self.train_y_mse):
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_mse = self.train_node_mse.compute() if self.train_node_mse.total > 0 else -1
        epoch_edge_mse = self.train_edge_mse.compute() if self.train_edge_mse.total > 0 else -1
        epoch_y_mse = self.train_y_mse.compute() if self.train_y_mse.total > 0 else -1

        to_log = {"train_epoch/epoch_X_mse": epoch_node_mse,
                  "train_epoch/epoch_E_mse": epoch_edge_mse,
                  "train_epoch/epoch_y_mse": epoch_y_mse}
        if wandb.run:
            wandb.log(to_log)
        return to_log



class TrainLossDiscrete(nn.Module):
    """ Train with Cross entropy"""
    def __init__(
        self, lambda_train,
        include_valence_loss=False,
        include_hybridization_loss=False,
    ):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_train = lambda_train
        self.include_valence_loss = include_valence_loss
        self.include_hybridization_loss = include_hybridization_loss

    def forward(self, masked_pred_X, masked_pred_E, pred_y, true_X, true_E, true_y, log: bool):
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean. """
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        loss_y = self.y_loss(pred_y, true_y) if true_y.numel() > 0 else 0.0

        loss_valence = 0.0
        loss_hybridization = 0.0
        if self.include_valence_loss:
            loss_valence = self.calculate_valence_penalty(flat_pred_X, flat_pred_E)
        if self.include_hybridization_loss:
            loss_hybridization = self.calculate_hybridization_penalty(flat_true_X, flat_true_E, flat_pred_X, flat_pred_E)

        total_loss = loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y
        total_loss += loss_valence + loss_hybridization

        if log:
            to_log = {
                "train_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
                "train_loss/X_CE": self.node_loss.compute() if true_X.numel() > 0 else -1,
                "train_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1,
                "train_loss/y_CE": self.y_loss.compute() if true_y.numel() > 0 else -1,
                'train_loss/valence': loss_valence,
                'train_loss/hybridization': loss_hybridization,
            }
            if wandb.run:
                wandb.log(to_log, commit=True)

        return total_loss

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        # epoch_y_loss = self.train_y_loss.compute() if self.y_loss.total_samples > 0 else -1
        epoch_y_loss = self.y_loss.compute() if self.y_loss.total_samples > 0 else -1

        to_log = {"train_epoch/x_CE": epoch_node_loss,
                  "train_epoch/E_CE": epoch_edge_loss,
                  "train_epoch/y_CE": epoch_y_loss}
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log


    def calculate_valence_penalty(self, pred_X, pred_E):
        # Define the expected valences for common atom types
        valences = {
            0: 4,  # Carbon
            1: 3,  # Nitrogen
            2: 2,  # Oxygen
            3: 1,  # Fluorine
            4: 1,  # Hydrogen
        }

        # Get predicted atom types as indices of one-hot encoding
        pred_atom_types = torch.argmax(pred_X, dim=-1)

        valence_penalty = 0.0

        for i, atom_type in enumerate(pred_atom_types):
            atom_type = atom_type.item()
            expected_valence = valences.get(atom_type, None)
            
            if expected_valence is not None:
                # Sum of bond orders for each atom
                actual_valence = torch.sum(pred_E[i])  # Total bond order for atom i
                
                # Penalize if valence exceeds expected or if any negative bond orders exist
                valence_excess_penalty = F.relu(actual_valence - expected_valence)
                negative_bond_penalty = F.relu(-pred_E[i]).sum()  # Sum of negative values if any

                # Aggregate penalties
                valence_penalty += (valence_excess_penalty + negative_bond_penalty)

        # Scale the total penalty
        return valence_penalty

    def calculate_hybridization_penalty(self, true_X, true_E, pred_X, pred_E):
        true_atom_types = torch.argmax(true_X, dim=-1)
        hybridization_penalty = 0.0
        for i, atom_type_tensor in enumerate(true_atom_types):
            atom_type = atom_type_tensor.item()  # Convert tensor to scalar for each atom type
            true_bond_orders = true_E[i].tolist()  # Actual bond orders
            expected_hybridization = self.infer_hybridization(atom_type, true_bond_orders)

            pred_bond_orders = pred_E[i].tolist()  # Predicted bond orders
            actual_hybridization = self.infer_hybridization(atom_type, pred_bond_orders)

            if actual_hybridization != expected_hybridization:
                hybridization_penalty += 1

        return hybridization_penalty


    def infer_hybridization(self, atom_type, bond_orders):
        """
        Infers the hybridization state of an atom based on its atom type and bond orders.

        Parameters:
        - atom_type: int, representing the atom type (e.g., 0 for Carbon)
        - bond_orders: list of floats, representing the bond orders to neighboring atoms

        Returns:
        - A string representing the hybridization state ('sp', 'sp2', 'sp3', etc.), or None if unknown.
        """
        # Define the number of valence electrons for common atom types
        valence_electrons = {
            0: 4,  # Carbon
            1: 5,  # Nitrogen
            2: 6,  # Oxygen
            3: 7,  # Fluorine
            4: 1,  # Hydrogen
        }

        # Get the total bond order (sum of bond orders)
        total_bond_order = sum(bond_orders)

        # Get the number of valence electrons for the atom
        valence = valence_electrons.get(atom_type)
        if valence is None:
            return None  # Unknown atom type

        # Approximate the number of sigma bonds
        # Each bond contributes one sigma bond, regardless of bond order
        sigma_bonds = len([b for b in bond_orders if b > 0])

        # Approximate the number of lone pairs
        # Total electrons used in bonding (each bond uses two electrons)
        bonding_electrons = total_bond_order
        lone_pair_electrons = valence - bonding_electrons
        # Each lone pair has two electrons
        lone_pairs = lone_pair_electrons / 2

        # Steric number = sigma bonds + lone pairs
        steric_number = sigma_bonds + lone_pairs

        # Use thresholds to account for non-integer bond orders and lone pairs
        # Determine hybridization based on steric number
        if 1.5 <= steric_number < 2.5:
            return 'sp'
        elif 2.5 <= steric_number < 3.5:
            return 'sp2'
        elif 3.5 <= steric_number < 4.5:
            return 'sp3'
        else:
            return None  # Hybridization states beyond sp3 are less common for these atoms