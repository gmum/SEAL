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


class SEALAtom(torch.nn.Module):
    """SEAL Network."""

    def __init__(self, kwargs):
        super(SEALAtom, self).__init__()
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
        self.input_features = kwargs.get("input_features", 21)
        self.hidden_features = kwargs.get("hidden_features", 64)
        self.num_layers = kwargs.get("num_layers", 3)
        self.regularization = kwargs.get("regularize", 0.0)
        self.regularization_contribution = kwargs.get("regularize_contribution", 0.0)
        self.dropout = torch.nn.Dropout(kwargs.get("drop", 0.))
        
        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.gcn_layers.append(
            SEALConv(self.input_features, self.hidden_features))
        self.batch_norms.append(nn.LayerNorm(self.hidden_features))

        for _ in range(self.num_layers-1):
            self.gcn_layers.append(
                SEALConv(self.hidden_features, self.hidden_features))
            self.batch_norms.append(nn.LayerNorm(self.hidden_features))

        self.linear = torch.nn.Linear(self.hidden_features, 1, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

        self.bn=nn.LayerNorm(self.hidden_features)
        
        
    def forward(self, data, mask_idx=None):
        """
        Args:
            data : The input data containing node features, edge indices, and other information.
            mask_idx (Optional[Tensor], optional): A mask to specify which part of graph need to be masked at contribution level. Defaults to None.
        """
        # Extract node features, adjacency, batch, and node assignment to clusters.
        # print(data)
        x, edge_index, s_node, batch, mask_breaks = data.x, data.edge_index, data.s_node, data.batch, data.mask_atom
        edge_brics_mask = mask_breaks.bool()

        # Apply GCN layers.
        for i, (conv, bn) in enumerate(zip(self.gcn_layers, self.batch_norms)):
            x = conv(x, edge_index,edge_brics_mask)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        # Convert to dense format.
        x_dense, mask = to_dense_batch(x, batch)
        adj_dense = to_dense_adj(edge_index, batch)
        s_dense, mask_s = to_dense_batch(s_node, batch)
        batch_size, num_nodes, _ = x_dense.size()

        # Apply masks
        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            mask_s = mask_s.view(batch_size, num_nodes, 1).to(s_node.dtype)
            x_dense, s_dense = x_dense * mask, s_dense * mask_s

        # Aggregate features from clusters.
        out = torch.matmul(s_dense.transpose(1, 2), x_dense)

        out = self.bn(out)

        # Compute cluster contribution.
        x_cluster_transformed = self.linear(out)
        
        x_cluster_transformed_mask=(s_dense.sum(dim=1, keepdim=True) > 0).transpose(1,2)
        x_cluster_transformed = x_cluster_transformed * x_cluster_transformed_mask
        if mask_idx is not None:
            x_cluster_transformed = x_cluster_transformed[:,mask_idx,:]

        # Sum contributions from all clusters.
        out = x_cluster_transformed.sum(dim=1)

        reg_loss = 0.
        bias_loss = 0.
        for layer in self.gcn_layers:
            reg_loss += torch.norm(layer.weights_seal_outside(), p=1)
            bias_loss += torch.norm(layer.bias_seal_outside(), p=1)

        # Apply bias
        out = out+self.bias
        l1_loss = abs(x_cluster_transformed).sum(dim=1) / ((torch.sum(s_dense.transpose(1, 2),
                                                                     dim=-1, keepdim=True) > 0) + 1e-7).sum(1)

        l1_loss = l1_loss.mean()
        # print("XD")
        return {"output": out, "losses": self.regularization*(reg_loss+bias_loss) + self.regularization_contribution * l1_loss , "s": s_node, "x_cluster_transformed": x_cluster_transformed}