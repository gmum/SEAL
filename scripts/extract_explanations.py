import argparse
import copy
import json
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import sys
from pathlib import Path
from datasetss.dataset import GraphFeaturizer, build_dataset, SyntheticGraphFeaturizer,SYNTHETIC_DATASET, STATS_DATASET
import sys
from architectures import SEALNetwork

def args_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--seed", type=int, default=123, help="seed")
    parser.add_argument("--save_path", type=str,required=True, help="path to save explanations")
    parser.add_argument("--explainer_path", type=str, required=True, help="path to explainer")
    args = parser.parse_args()
    return args


def create_model(kwargs: dict) -> torch.nn.Module:
    """
    Create a model instance based on the provided arguments.
    Args:
        kwargs (dict): Arguments to configure the model.

    Returns:
        torch.nn.Module: The created model instance.
    """
    models = {
        "SEAL": SEALNetwork,
    }
    model_name = kwargs.get("model", "")
    if model_name not in models:
        raise ValueError(f"Unknown model name: {model_name}")

    return models[model_name](kwargs)

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
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)


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
        "split": loaded['args']['split']
    }


    _, _, dataset_test = build_dataset(dataset_kwargs)

    featurizer=GraphFeaturizer(y_column='Y') if not synthetic else  SyntheticGraphFeaturizer(y_column='Y')

    test_set = featurizer(dataset_test, dataset_kwargs)
    dataloader_test = DataLoader(test_set, batch_size=1, shuffle=False)

    model_kwargs = {
        "model": "SEAL",
        "hidden_features": loaded['model_args']['hidden_features'],
        "input_features": test_set[0].x.shape[1],
        "drop": 0.0,
        "num_layers": loaded['model_args']['num_layers'],
        "task": loaded_args["task"],
        "number_of_clusters": 0,
        "regularize": 0,
        "weight_decay": 0,
    }
    explainer = create_model(model_kwargs)
    explainer.load_state_dict(loaded["state_dict"])
    explainer = explainer.to(device)
    explainer.eval()



    explanations = list()
    for batch in tqdm(dataloader_test):
        batch = batch.to(device)
        node_mask = list()

        try:
            out = explainer(batch, None)
        except Exception as e:
            print(f"Error occurred while explaining batch: {e}")
            continue
        
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
    if loaded_args["task"] == "classification":
        torch.save(
            {
                "args": vars(args),
                "model_args": loaded_args,
                "explanations": explanations,
                "f1": loaded["f1"],
                "roc_auc": loaded["roc_auc"],
                "accuracy":loaded["accuracy"],
                "explainer": model_kwargs ,
            },
            args.save_path,
        )
    else:
        torch.save(
            {
                "args": vars(args),
                "model_args": loaded_args,
                "explanations": explanations,
                "mae": loaded["mae"],
                "rmse": loaded["rmse"],
                "explainer": model_kwargs,
            },
            args.save_path,
        )
    print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    main()