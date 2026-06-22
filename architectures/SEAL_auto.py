import torch
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch.nn import functional as F
from torch import nn
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, Tensor
from typing import Union, Tuple, List, Optional
from torch.nn import Linear
from torch_geometric.nn.aggr import Aggregation
from .SEAL import SEALConv
from torch_geometric.nn import GCNConv, dense_diff_pool



class SEALAuto(torch.nn.Module):
    """SEAL Network."""

    def __init__(self, kwargs):
        super(SEALAuto, self).__init__()
        """Initializes the SEALNetwork model.
        Args:
            kwargs (dict): Dictionary containing the following
                parameters:
                - input_features (int): Number of input features.
                - hidden_features (int): Number of hidden features.
                - num_layers (int): Number of layers.
                - drop (float): Dropout
                - regularize (float): Regularization term for message passing.
                - regularize_contribution (float): Regularization term for cluster contributions.
        """
        self.input_features = kwargs.get("input_features", 64)
        self.hidden_features = kwargs.get("hidden_features", 64)
        self.num_layers = kwargs.get("num_layers", 3)
        self.input_features = kwargs.get("input_features", 64)
        self.hidden_features = kwargs.get("hidden_features", 64)
        self.num_layers = kwargs.get("num_layers", 3)
        self.num_clusters = kwargs.get("number_of_clusters", 5)

        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First GCN layer
        self.gcn_layers.append(
            GCNConv(self.input_features, self.hidden_features))
        self.batch_norms.append(nn.BatchNorm1d(self.hidden_features))

        # Hidden GCN layers
        for _ in range(self.num_layers-1):
            self.gcn_layers.append(
                GCNConv(self.hidden_features, self.hidden_features))
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_features))

        # Final linear layer for classification/regression
        self.conv_pool = GCNConv(self.hidden_features, self.num_clusters)

        self.linear = torch.nn.Linear(self.hidden_features, 1)
        
        self.loss_lp=0.1
        self.loss_h=0.9
        
    def forward(self, data):
        """
        Args:
            data : The input data containing node features, edge indices, and other information.
        """
        # Extract node features, adjacency, batch, and node assignment to clusters.
        # print(data)
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, bn in zip(self.gcn_layers, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)  # Apply batch normalization
            x = F.relu(x)

        s = self.conv_pool(x, edge_index)

        # Convert the node features and adjacency to dense format for diff pooling.
        x_dense, mask = to_dense_batch(x, batch)
        adj_dense = to_dense_adj(edge_index, batch)
        s_dense, mask_s = to_dense_batch(s, batch)

        x_pool, adj_pool, loss_lp, loss_h = dense_diff_pool(
            x_dense, adj_dense, s_dense, mask)

        x_cluster_transformed = self.linear(x_pool)
        # print(x_cluster_transformed)
        out = x_cluster_transformed.sum(dim=1)
        loss= self.loss_lp*loss_lp + self.loss_h*loss_h
        return {"output": out, "losses":loss, "s": s_dense, "x_cluster_transformed": x_cluster_transformed}
