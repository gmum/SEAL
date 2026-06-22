import argparse
import copy
import json
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Subset

from torch_geometric.data import Data, Batch
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, subgraph
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn import GCN, GAT, GIN, global_add_pool
from torch_geometric.nn import Sequential, Linear
import sys
import random


from sklearn.metrics import roc_auc_score
from datasetss import SyntheticGraphFeaturizer, build_dataset, MutagGraphFeaturizer, SYNTHETIC_DATASET, STATS_DATASET, MUTAG_DATASET
from architectures import SEALNetwork
# from d
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def args_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--seed", type=int, default=123, help="seed")
    parser.add_argument("--save_path", type=str,required=True, help="path to save explanations")
    parser.add_argument("--explainer_path", type=str, required=True, help="path to explainer")
    args = parser.parse_args()
    return args

def create_model(kwargs):
    return SEALNetwork(kwargs)

def get_node_importance(e:torch.Tensor)->torch.Tensor:
    """
    Get the importance scores of nodes from the explanation tensor.
    Args:
        e (torch.Tensor): Explanation tensor containing node importance scores.
    Returns:
        torch.Tensor: A tensor containing the importance scores of the nodes and in case of feature attribution, the importance scores of the features are aggregated.
    """
    node_mask = e.node_mask.detach() if hasattr(e, "node_mask") else None
    node_mask = torch.nan_to_num(node_mask)
    if len(node_mask.shape) == 2:
        node_mask = node_mask.sum(-1)
    assert len(node_mask.shape) == 1
    return node_mask


def main():
    args = args_parser()
    print(args)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded = torch.load(args.explainer_path, map_location=torch.device("cpu"), weights_only=False)
    loaded_args = loaded["args"]

    synthetic = loaded['args']['data_set'] in SYNTHETIC_DATASET
    mean=STATS_DATASET.get(loaded['args']['data_set'], {}).get("mean", 0.0)
    std=STATS_DATASET.get(loaded['args']['data_set'], {}).get("std", 1.0)

    dataset_kwargs = {
        "data_set": loaded['args']['data_set'],
        "mean": mean,
        "std": std,
        "y_column": loaded['args']['Y_column'],
        "smiles_col": "Drug",
        "task": "classification",
        "split": loaded_args["split"] if "split" in loaded_args else 0,
        }
    print(dataset_kwargs)
    # return
    _, _, dataset_test = build_dataset(dataset_kwargs)
    print(f"Test dataset: {len(dataset_test)}")
    # featurizer=MutagGraphFeaturizer(y_column='Y')
    featurizer=MutagGraphFeaturizer(y_column='Y') if loaded_args['data_set'] in MUTAG_DATASET else SyntheticGraphFeaturizer(y_column='Y')
    print(f"Featurizer: {len(dataset_test)}")

    test_set = featurizer(dataset_test, dataset_kwargs)

    dataloader_test = DataLoader(test_set, batch_size=1, shuffle=False)
    model_weights = torch.load(
    args.explainer_path, map_location=torch.device(device))["state_dict"]

    model_kwargs = {
        "model": "SEAL",
        "hidden_features": loaded_args["hidden_dim"],
        "input_features": test_set[0].x.shape[1],
        "drop": 0.0,
        "num_layers": loaded_args["num_layers"],
        "task": loaded_args["task"],
        "number_of_clusters": 0,
        "regularize": 0,
        "weight_decay": 0,
    }
    explainer = create_model(model_kwargs)
    explainer.load_state_dict(model_weights)
    explainer = explainer.to(device)
    explainer.eval()

    explanations = list()
    for batch in tqdm(dataloader_test):
        batch = batch.to(device)
        node_mask = list()

        out = explainer(batch, None)
        mask=torch.zeros(batch.x.shape[0], device=device)
        for i in range(batch.s.shape[0]):
            mask[i] = out["x_cluster_transformed"][0][batch.s[i].argmax().detach().cpu().item()]
        node_mask = mask
        pred=out["output"]
        num_nodes = 0
        for b in range(batch.batch.max() + 1):
            x_mask = batch.batch == b
            edge_index_mask = batch.batch[batch.edge_index[0]] == b
            edge_index = batch.edge_index[:, edge_index_mask] - num_nodes
            data = Data(
                x=batch.x[x_mask].detach().cpu(),
                edge_index=edge_index.detach().cpu(),
                y=batch.y[b].detach().cpu(),
                pred=pred[b].detach().cpu(),
                node_mask=node_mask.detach().cpu(),
            )

            explanations.append(data)
            num_nodes += x_mask.sum()

    print(f"Finished with {len(explanations)}/{len(dataset_test)} explanations")
    torch.save(
        {
            "args": vars(args),
            "model_args": loaded_args,
            "explanations": explanations,
            "explainer": model_kwargs,
        },
        args.save_path,
    )
    print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    main()
