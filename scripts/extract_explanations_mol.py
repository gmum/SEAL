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
import sys
import random


from datasetss import SyntheticGraphFeaturizer, build_dataset

from datasetss import GraphFeaturizer, build_dataset
from architectures import SEALNetwork, SEALAtom



EXPLAINERS = [
    "SEAL",
    "SEALAtom",
]

def args_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--seed", type=int, default=123, help="seed")
    parser.add_argument("--save_path", type=str,default="/home/sebek/sebek/bxaic/new/cyp/Saliency_explain_cyp_0.pt" ,required=False, help="path to save explanations")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--explainer_path", type=str, required=False,default="/home/sebek/sebek/bxaic/weights/weights/cyp/model_GIN_cyp_0.pt", help="model path")
    parser.add_argument(
        "--explainer_type",
        type=str,
        required=False,
        default="SEAL",
        choices=EXPLAINERS,
        help="explainer type",
    )
    args = parser.parse_args()
    return args


def get_node_importance(node_mask):
    node_mask = node_mask.detach() 
    if len(node_mask.shape) == 2:
        node_mask = node_mask.sum(-1)
    assert len(node_mask.shape) == 1
    return node_mask


def create_model(kwargs):
    models = {
        "SEAL": SEALNetwork,
        "SEALAtom": SEALAtom,
    }
    model_name = kwargs.get("model", "")
    if model_name not in models:
        raise ValueError(f"Unknown model name: {model_name}")


    return models[model_name](kwargs)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def main():
    args = args_parser()

    print(args)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded = torch.load(args.explainer_path, map_location=torch.device("cpu"), weights_only=False)
    loaded_args = loaded["args"]
    # print(loaded)
    # print(loaded_args)
    if loaded['args']['data_set'] == "sol":
        dataset_kwargs = {
            "data_set": loaded_args["data_set"],
            "mean": -2.86,
            "std": 2.38,
            "y_column": 'Y',
            "smiles_col": "Drug",
            "task": loaded_args["task"],
            "split": loaded_args["split"] if "split" in loaded_args else 0,
        }
    elif loaded['args']['data_set'] == "sol_s":
        dataset_kwargs = {
            "data_set": loaded_args["data_set"],
            "mean": -2.51,
            "std": 2.20,
            "y_column": 'Y',
            "smiles_col": "Drug",
            "task": loaded_args["task"],
            "split": loaded_args["split"] if "split" in loaded_args else 0,
        }
    else:
        dataset_kwargs = {
            "data_set": loaded_args["data_set"],
            "mean": 0.0,
            "std": 1.0,
            "y_column": 'Y',
            "smiles_col": "Drug",
            "task": loaded_args["task"],
            "split": loaded_args["split"] if "split" in loaded_args else 0,
        }


    _, _, dataset_test = build_dataset(dataset_kwargs)
    featurizer=GraphFeaturizer(y_column='Y')
    
    test_set = featurizer(dataset_test, dataset_kwargs)
    dataloader_test = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model_kwargs = {
        "model": args.explainer_type,
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
    explainer.load_state_dict(loaded["state_dict"])
    explainer = explainer.to(device)
    explainer.eval()

             
    explanations = list()
    for batch in tqdm(dataloader_test):
        batch = batch.to(device)
        target = batch.y
        node_mask = list()
        
        if args.explainer_type == "SEAL":
            out = explainer(batch, None)
            mask=torch.zeros(batch.x.shape[0], device=device)
            for i in range(batch.s.shape[0]):
                mask[i] = out["x_cluster_transformed"][0][batch.s[i].argmax().detach().cpu().item()]
            node_mask = mask
        elif args.explainer_type == "SEALAtom":
            out = explainer(batch, None)
            mask=torch.zeros(batch.x.shape[0], device=device)
            for i in range(batch.s_node.shape[0]):
                mask[i] = out["x_cluster_transformed"][0][batch.s_node[i].argmax().detach().cpu().item()]
            node_mask = mask
        else:
            # For other explainers, you can implement their logic here.
            raise NotImplementedError(f"Explainer type {args.explainer_type} is not implemented.")
        node_mask = mask
        num_nodes = 0
        for b in range(batch.batch.max() + 1):
            x_mask = batch.batch == b
            edge_index_mask = batch.batch[batch.edge_index[0]] == b
            edge_index = batch.edge_index[:, edge_index_mask] - num_nodes
            data = Data(
                x=batch.x[x_mask].detach().cpu(),
                edge_index=edge_index.detach().cpu(),
                y=batch.y[b].detach().cpu(),
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
            "explainer": model_kwargs if args.explainer_type == "SEAL" else {},
        },
        args.save_path,
    )
    print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    main()
