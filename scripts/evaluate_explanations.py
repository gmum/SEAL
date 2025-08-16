import argparse
import copy
import json
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import csv
import os
import torch
from torch.utils.data import Subset

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, subgraph
from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.metrics import f1_score, mean_squared_error as mse_score
import sys
from pathlib import Path
from datasetss.dataset import SYNTHETIC_DATASET, CLASSIFICATION_DATASET,REGRESSION_DATASET, STATS_DATASET

from datasetss.dataset import GraphFeaturizer, build_dataset, SyntheticGraphFeaturizer
from architectures import SEALNetwork

def no_outliers(x: torch.Tensor)-> bool:
    """
    Check if a tensor contains outliers based on the IQR method.
    Adapted from: https://github.com/mproszewska/B-XAIC/blob/main/evaluate_explanations.py
    Args:
        x (torch.Tensor): Input tensor to check for outliers.

    Returns:
        bool: True if no outliers are present, False otherwise.
    """
    Q1 = torch.quantile(x, 0.25)
    Q3 = torch.quantile(x, 0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = x[(x < lower_bound) | (x > upper_bound)]
    return float(len(outliers) == 0)



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

def args_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--seed", type=int, default=123, help="seed")
    parser.add_argument("--save_path", type=str,required=True, help="path to csv save results")
    parser.add_argument("--explanations_path", type=str, required=True, help="path to explanations")
    parser.add_argument("--percentage", type=float, default=0.1, help="percentage of nodes to mask")
    args = parser.parse_args()
    return args

def main():
    args = args_parser()
    print(args)
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    loaded = torch.load(args.explanations_path, weights_only=False)
    
    synthetic = loaded['model_args']['data_set'] in SYNTHETIC_DATASET
    mean=STATS_DATASET.get(args.data_set, {}).get("mean", 0.0)
    std=STATS_DATASET.get(args.data_set, {}).get("std", 1.0)

    dataset_kwargs = {
        "data_set": args.data_set,
        "mean": mean,
        "std": std,
        "y_column": args.Y_column,
        "smiles_col": "Drug",
        "split": args.split
    }
    _,_, dataset_test = build_dataset(dataset_kwargs)
    featurizer=GraphFeaturizer(y_column='Y') if not synthetic else  SyntheticGraphFeaturizer(y_column='Y')

    test_set = featurizer(dataset_test, dataset_kwargs) 
    dataloader_test = DataLoader(test_set, batch_size=1, shuffle=False)
    explanations = loaded["explanations"]

    if synthetic:
        metrics=eval_synth(explanations, dataloader_test, loaded, args)
    else:
        metrics=eval_real(explanations, dataloader_test, loaded, args)

    if args.save_path is not None:
        file_exists = os.path.exists(args.save_path)
        with open(args.save_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

def eval_real(explanations: torch.Tensor, dataloader_test: DataLoader, loaded: dict, args)->dict:
    """
    Evaluate the model on the real dataset using the provided explanations.

    Args:
        explanations (torch.Tensor): Explanation tensor containing node importance scores.
        dataloader_test (DataLoader): DataLoader for the test dataset.
        loaded (dict): Loaded model and dataset information.
        args: Argument parser containing command line arguments.
    Returns:
        dict: Evaluation metrics for the real dataset.
    """

    non_abs=False
    mask_contr=False

    pos_fidelity_models, neg_fidelity_models = list(), list()
    task=loaded['model_args']['task']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_weights = torch.load(loaded["args"]["explainer_path"], map_location=torch.device(device))['state_dict']
    model_kwargs = loaded["explainer"]
    model = create_model(model_kwargs)
    model.load_state_dict(model_weights)
    model = model.to(device)
    model.eval()
    # accs=[]
    s=[]
    for e, data in tqdm(zip(explanations, dataloader_test), total=len(explanations)):
        assert torch.equal(e.x, data.x)
        data = data.to(device)
        node_mask = e["node_mask"]
        # abs_mask = torch.abs(node_mask)
        if non_abs:
            abs_mask = node_mask
        # abs_mask=node_mask
        else:
            abs_mask = torch.abs(node_mask)
        original_data = copy.deepcopy(data)
        original_output = model(data, mask_idx=None)["output"]
        i=1
        while True:
            tolerance = 1e-3            
            if non_abs:
                if original_output >= 0:
                    max_val = torch.topk(torch.unique(abs_mask), i)[0]
                else:
                    max_val = torch.topk(torch.unique(abs_mask), i, largest=False)[0]
            else:
                max_val = torch.topk(torch.unique(abs_mask), i)[0]
            diff = torch.abs(abs_mask.unsqueeze(-1) - max_val) 
            is_close = (diff < tolerance).any(dim=1)            
            topk_indices = is_close.nonzero(as_tuple=True)[0]                
            if len(topk_indices)/ data.x.shape[0] >= args.percentage:
                break
            i += 1            
        
        s.append(len(topk_indices))
        top_k_clusters = torch.unique(data.s[topk_indices].argmax(-1)).detach().cpu().tolist()
        # Create mask
        mask = torch.zeros_like(node_mask, dtype=torch.bool, device=device)
        mask[topk_indices] = True

        mask = mask.to(device)
        
        inverse_mask = ~mask
        inverse_mask = inverse_mask.to(device)
        data_with_mask = copy.deepcopy(data)
        data_with_mask.x = data_with_mask.x * mask.unsqueeze(-1)
        
        data_with_inverse_mask = copy.deepcopy(data)
        data_with_inverse_mask.x = data_with_inverse_mask.x * inverse_mask.unsqueeze(-1)
        
        if mask_contr:
            masked_output = model(data_with_mask, mask_idx=None)["output"]
            inverse_masked_output = model(data_with_inverse_mask, mask_idx=None)["output"]
        else:
            masked_output = model(data_with_mask, mask_idx=top_k_clusters)["output"]
            inverse_masked_output = model(data_with_inverse_mask, mask_idx=list(set(range(data.num_cluster.item())) - set(top_k_clusters)))["output"]
        
        if task == "classification":
            original_output = original_output>=0
            masked_output = masked_output>=0
            inverse_masked_output = inverse_masked_output>=0            
            pos_fidelity_model= (inverse_masked_output == original_output).float().mean()
            neg_fidelity_model = (masked_output == original_output).float().mean()
            
        else:
            original_output = original_output.squeeze()
            masked_output = masked_output.squeeze()
            inverse_masked_output = inverse_masked_output.squeeze()
            pos_fidelity_model = torch.abs(inverse_masked_output - original_output)
            neg_fidelity_model = torch.abs(masked_output - original_output)

        pos_fidelity_models.append(pos_fidelity_model.item())
        neg_fidelity_models.append(neg_fidelity_model.item())

    if task == "classification":
        pos_fidelity_model=1-torch.tensor(pos_fidelity_models).mean().item()
        neg_fidelity_model=1-torch.tensor(neg_fidelity_models).mean().item()
    else:
        pos_fidelity_model=torch.tensor(pos_fidelity_models).mean().item()
        neg_fidelity_model=torch.tensor(neg_fidelity_models).mean().item()

    metrics = {
        "data_set": loaded["model_args"]["data_set"],
        "explainer_type": loaded["args"]["explainer_type"],
        "split": loaded["model_args"]["split"],
        "pos_fidelity_models": pos_fidelity_model,
        "neg_fidelity_models": neg_fidelity_model,
        "percentage": args.percentage,
    }
    print(f"Positive Fidelity Model: {metrics['pos_fidelity_models']}")
    print(f"Negative Fidelity Model: {metrics['neg_fidelity_models']}")
    return metrics


def eval_synth(explanations: torch.Tensor, dataloader_test: DataLoader, loaded: dict, args) -> dict:
    """Evaluate the model on the synthetic dataset using the provided explanations.

    Args:
        explanations (torch.Tensor): Explanation tensor containing node importance scores.
        dataloader_test (DataLoader): DataLoader for the test dataset.
        loaded (dict): Loaded model and arguments.
        args: Command line arguments.
    Returns:
        dict: Evaluation metrics for the synthetic dataset.
    """
    non_empty_ex, empty_ex = list(), list()
    metric_fn = roc_auc_score
    print(f"Task: {loaded['model_args']['task']} Eval metric: {metric_fn.__name__}")

    non_empty_ex, empty_ex = list(), list()

    for e, data in tqdm(zip(explanations, dataloader_test), total=len(explanations)):
        assert torch.equal(data.x, e.x) and torch.equal(data.edge_index, e.edge_index), "Data and explanation do not match"
        assert data.y.item()==e.y.item(), "Data and explanation do not match"
        assert data.explanation.reshape(-1).shape[0] == e.node_mask.reshape(-1).shape[0], "Data and explanation do not match"
        pred_mask = get_node_importance(e)
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
        "f1": loaded["f1"],
        "roc_auc": loaded["roc_auc"],
        "accuracy": loaded["accuracy"],
        "split": loaded["model_args"]["split"],
    }
    print(f"Non empty explanations: {metrics['non_empty_ex']}")
    print(f"Empty explanations: {metrics['empty_ex']}")

    return metrics
     

if __name__ == "__main__":
    main()