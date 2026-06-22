import argparse
import copy
import json
import os
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Subset
from datasetss import SyntheticGraphFeaturizer, build_dataset, MutagGraphFeaturizer, SYNTHETIC_DATASET, STATS_DATASET, MUTAG_DATASET

from torch_geometric.data import Data, Batch
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, subgraph

from sklearn.metrics import roc_auc_score, average_precision_score
from extract_explanations import create_model
import random
from torch_geometric.nn import MessagePassing


def args_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--seed", type=int, default=123, help="seed")
    parser.add_argument("--save_path", type=str, default="/results.csv", help="path to save results")
    parser.add_argument("--explanations_path", type=str, required=False, default="path.pt")
    args = parser.parse_args()
    return args


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def get_node_importance(e, data):
    node_mask = e.node_mask.detach() if hasattr(e, "node_mask") else None
    node_mask = torch.nan_to_num(node_mask)
    if len(node_mask.shape) == 2:
        node_mask = node_mask.sum(-1)
    assert len(node_mask.shape) == 1
    return node_mask



def no_outliers(x):
    Q1 = torch.quantile(x, 0.25)
    Q3 = torch.quantile(x, 0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = x[(x < lower_bound) | (x > upper_bound)]
    return float(len(outliers) == 0)


def main():
    args = args_parser()
    print(args)
    set_seed(args.seed)
    loaded = torch.load(args.explanations_path, weights_only=False)
    task = loaded["model_args"]["task"]
    print(loaded["model_args"])

    dataset_kwargs = {
        "data_set": loaded["model_args"]["data_set"],
        "mean": 0.0,
        "std": 1.0,
        "y_column": 'Y',
        "smiles_col": "Drug",
        "task": "classification",
        "split": loaded["model_args"]["split"],
    }
    print(f"Dataset args: {dataset_kwargs}")
    _,_, dataset_test = build_dataset(dataset_kwargs)
    featurizer=MutagGraphFeaturizer(y_column='Y') if loaded["model_args"]["data_set"] in MUTAG_DATASET else SyntheticGraphFeaturizer(y_column='Y')

    test_set = featurizer(dataset_test, dataset_kwargs)
    dataloader_test = DataLoader(test_set, batch_size=1, shuffle=False)
    explanations = loaded["explanations"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    non_empty_ex, empty_ex = list(), list()
    spars = list()
    metric_fn = roc_auc_score
    print(f"Task: {task} Eval metric: {metric_fn.__name__}")

    non_empty_ex, empty_ex = list(), list()
    idx_to_remove = 0
    if len(explanations) != len(dataloader_test):
        print(f"Warning: Number of explanations ({len(explanations)}) does not match number of test samples ({len(dataloader_test)})")
        for i in range(len(explanations)):
            if torch.equal(test_set[i].x, explanations[i].x) and torch.equal(test_set[i].edge_index, explanations[i].edge_index):
                idx_to_remove += 1
                continue
            else:
                print(f"Removing explanation at index {i} due to mismatch")
                explanations.pop(i)
                break
                
    for e, data in tqdm(zip(explanations, dataloader_test), total=len(explanations)):
        assert torch.equal(data.x, e.x) and torch.equal(data.edge_index, e.edge_index), "Data and explanation do not match"
        assert data.y.item()==e.y.item(), "Data and explanation do not match"
        assert data.explanation.reshape(-1).shape[0] == e.node_mask.reshape(-1).shape[0], "Data and explanation do not match"
        pred_mask = get_node_importance(e, data)
        gt = data.explanation

        if gt.min() == gt.max():
            m = no_outliers(pred_mask)
            empty_ex.append(m)
        else:
            m = metric_fn(gt, pred_mask)
            non_empty_ex.append(m)
            
    metrics = {
        "non_empty_ex": torch.tensor(non_empty_ex).mean().item(),
        "empty_ex": torch.tensor(empty_ex).mean().item(),
        "split": loaded["model_args"]["split"],
    }

    print(f"'non_empty_ex': '{metrics['non_empty_ex']:.4f}', 'empty_ex': '{metrics['empty_ex']:.4f}'")
    import csv
    if args.save_path is not None:
        file_exists = os.path.exists(args.save_path)
        with open(args.save_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)


if __name__ == "__main__":
    main()