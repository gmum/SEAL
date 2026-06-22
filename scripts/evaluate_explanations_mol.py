import argparse
import copy
import json
import os
from pyexpat import model
from turtle import pos
from tqdm import tqdm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv import GINConv

import numpy as np
import pandas as pd
import csv
import os
import torch
from torch.utils.data import Subset

from torch_geometric.data import Data, Batch
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, subgraph

from sklearn.metrics import f1_score, mean_squared_error as mse_score
from extract_explanations_mol import create_model

from datasetss import GraphFeaturizer, build_dataset
from trainGNNs import get_model, set_seed
import sys


from architectures import SEALNetwork, SEALAtom




def args_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--seed", type=int, default=123, help="seed")
    parser.add_argument("--save_path", type=str, default=None, help="path to save results")
    parser.add_argument("--explanations_path", type=str,default="/home/sebek/sebek/bxaic/debug/explanations_IntegratedGradients_sol_4.pt" ,required=False)
    parser.add_argument("--model_path", type=str, required=False,default="/home/sebek/sebek/bxaic/debug/model_GIN_sol_4.pt", help="model path")
    parser.add_argument("--percentage", type=float, default=0.3, help="percentage of nodes to mask")
    parser.add_argument("--absolute", type=int, help="use absolute values of attributions")
    parser.add_argument("--mask_contribution", type=int, help="mask contribution instead of attribution")
    args = parser.parse_args()
    return args

def main():
    args = args_parser()
    print(args)
    set_seed(args.seed)
    loaded_model = torch.load(args.model_path, map_location=torch.device("cpu"), weights_only=False)
    loaded = torch.load(args.explanations_path, weights_only=False)
    task = loaded_model["args"]["task"]
    loaded_args = loaded_model["args"]

    if loaded_args['data_set'] == "sol":
        dataset_kwargs = {
            "data_set": loaded_args["data_set"],
            "mean": -2.86,
            "std": 2.38,
            "y_column": 'Y',
            "smiles_col": "Drug",
            "task": loaded_args["task"],
            "split": loaded_args["split"] if "split" in loaded_args else 0,
        }
    elif loaded_args['data_set'] == "sol_s":
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
    _,_, dataset_test = build_dataset(dataset_kwargs)
    featurizer=GraphFeaturizer(y_column='Y')

    test_set = featurizer(dataset_test, dataset_kwargs)
    dataloader_test = DataLoader(test_set, batch_size=1, shuffle=False)
    explanations = loaded["explanations"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pos_fidelitys, neg_fidelitys, accs = list(), list(), list()
    metrics_ = list()
    print(loaded_args)
    print(loaded["args"])
    model_weights = torch.load(args.model_path, map_location=torch.device(device))["state_dict"]
    model_kwargs = {
        "model": loaded["args"]["explainer_type"],
        "hidden_features": loaded_args["hidden_dim"],
        "input_features": test_set[0].x.shape[1],
        "drop": 0.0,
        "num_layers": loaded_args["num_layers"],
        "task": loaded_args["task"],
        "number_of_clusters": 0,
        "regularize": 0,
        "weight_decay": 0,
    }
    # model_kwargs = loaded_args["explainer"]
    model = create_model(model_kwargs)
    model.load_state_dict(model_weights)
    model = model.to(device)

    model.eval()

    s=[]
    numbers_to_mask=[]
    for  ids, (e, data) in tqdm(enumerate(zip(explanations, dataloader_test)), total=len(explanations)):
        assert torch.equal(e.x, data.x)

        data = data.to(device)
        
        if loaded["args"]["explainer_type"]=="SEAL":
            node_mask = e["node_mask"]
            # abs_mask = torch.abs(node_mask) # TODO: abs
            abs_mask=node_mask # TODO: max value of original
            if args.absolute:
                abs_mask = torch.abs(node_mask)
            
            original_data = copy.deepcopy(data)
            original_output = model(data, mask_idx=None)["output"]

            # max_val = torch.max(abs_mask)
            i=1
            while True:
                tolerance = 1e-3
                # max_val is a tensor of shape [i]
                if args.absolute:
                    max_val = torch.topk(torch.unique(abs_mask), i)[0] # Just top i unique values
                # For negative predictions, we want the smallest values of the mask
                else: 
                    if original_output>=0:
                        max_val = torch.topk(torch.unique(abs_mask), i)[0]
                    else:
                        max_val = torch.topk(torch.unique(abs_mask), i, largest=False)[0]

                
                diff = torch.abs(abs_mask.unsqueeze(-1) - max_val) 
                is_close = (diff < tolerance).any(dim=1)            
                topk_indices = is_close.nonzero(as_tuple=True)[0]                
                if len(topk_indices)/ data.x.shape[0] >= args.percentage:
                    break
                i += 1            
            
            s.append(len(topk_indices))
            top_k_clusters = torch.unique(data.s[topk_indices].argmax(-1)).detach().cpu().tolist()

            numbers_to_mask.append(len(topk_indices))
            
            # Create mask
            mask = torch.zeros_like(node_mask, dtype=torch.bool, device=device)
            mask[topk_indices] = True

            mask = mask.to(device)
            # MASK
            #1 to most important
            #0 to least important

            # INVERSEMASK
            #0 to most important
            #1 to least important
            inverse_mask = ~mask
            inverse_mask = inverse_mask.to(device)
            original_data = copy.deepcopy(data)
            data_with_mask = copy.deepcopy(data)
            data_with_mask.x = data_with_mask.x * mask.unsqueeze(-1)
            #DATA WITH MASK HAS ONLY THE IMPORTANT NODES
            # THE REST ARE ZEROED OUT

            #DATA WITH INVERSE MASK HAS THE LEAST IMPORTANT NODES
            # THE IMPORTANT ONES ARE ZEROED OUT
            data_with_inverse_mask = copy.deepcopy(data)
            data_with_inverse_mask.x = data_with_inverse_mask.x * inverse_mask.unsqueeze(-1)
            
            original_output = model(data, mask_idx=None)["output"]
            if args.mask_contribution:
                masked_output = model(data_with_mask, mask_idx=top_k_clusters)["output"]
                inverse_masked_output = model(data_with_inverse_mask, mask_idx=list(set(range(data.num_cluster.item())) - set(top_k_clusters)))["output"]
            else:
                masked_output = model(data_with_mask, mask_idx=None)["output"]
                inverse_masked_output = model(data_with_inverse_mask, mask_idx=None)["output"]

            # MASKED OUTPUT as INPUT -> ONLY IMPORANT NODES: at the end we stay only with the important nodes
            # INVERSE MASKED OUTPUT as INPUT -> ONLY UNIMPORTANT NODES: at the end we stay only with the unimportant nodes
            if task == "classification":
                original_output = original_output>=0
                masked_output = masked_output>=0
                inverse_masked_output = inverse_masked_output>=0
                pos_fidelity = ((inverse_masked_output == original_output).float()).abs().mean()
                neg_fidelity = ((masked_output == original_output).float()).abs().mean()
                metric_ = (data.y == original_output).float().mean().item()
                
            else:
                original_output = original_output.squeeze()
                masked_output = masked_output.squeeze()
                inverse_masked_output = inverse_masked_output.squeeze()
                pos_fidelity = torch.abs(inverse_masked_output - original_output)
                neg_fidelity = torch.abs(masked_output - original_output)
                metric_= abs(data.y - original_output).float().mean().item()

        
        elif loaded["args"]["explainer_type"]=="SEALAtom":
            # abs_mask = node_mask  # TODO: max value of original
            # if args.absolute:
            #     abs_mask = torch.abs(node_mask)

            original_data = copy.deepcopy(data)
            original_output = model(data, mask_idx=None)["output"]

            number_masked = torch.load(f"/shared/results/common/z1185722/gnn/ids/{loaded_args['data_set']}_{args.percentage}_{loaded_args['split']}_{int(args.absolute)}_{int(args.mask_contribution)}.pt")

            number_to_mask = number_masked[ids].item()

            try:
                if args.absolute:
                    # Select nodes with highest absolute attribution values
                    topk_vals, topk_indices = torch.topk(torch.abs(e["node_mask"]), number_to_mask)
                else:
                    # For non-absolute: select based on sign of original prediction
                    if original_output >= 0:
                        # Positive prediction: select highest values
                        topk_vals, topk_indices = torch.topk(e["node_mask"], number_to_mask, largest=True)
                    else:
                        # Negative prediction: select smallest (most negative) values
                        topk_vals, topk_indices = torch.topk(e["node_mask"], number_to_mask, largest=False)
            except Exception as exp:
                topk_indices = torch.arange(e["node_mask"].shape[0], device=device)
                print("WARNING: Exception in topk:", exp)
            s.append(len(topk_indices))

            mask = torch.zeros_like(e["node_mask"], dtype=torch.bool, device=device)
            mask[topk_indices] = True
            top_k_clusters = torch.unique(data.s_node[topk_indices].argmax(-1)).detach().cpu().tolist()

            numbers_to_mask.append(len(topk_indices))
            
            # Create mask
            mask = mask.to(device)
            # MASK
            #1 to most important
            #0 to least important

            # INVERSEMASK
            #0 to most important
            #1 to least important
            inverse_mask = ~mask
            inverse_mask = inverse_mask.to(device)
            original_data = copy.deepcopy(data)
            data_with_mask = copy.deepcopy(data)
            data_with_mask.x = data_with_mask.x * mask.unsqueeze(-1)
            #DATA WITH MASK HAS ONLY THE IMPORTANT NODES
            # THE REST ARE ZEROED OUT

            #DATA WITH INVERSE MASK HAS THE LEAST IMPORTANT NODES
            # THE IMPORTANT ONES ARE ZEROED OUT
            data_with_inverse_mask = copy.deepcopy(data)
            data_with_inverse_mask.x = data_with_inverse_mask.x * inverse_mask.unsqueeze(-1)
            
            original_output = model(data, mask_idx=None)["output"]
            if args.mask_contribution:
                masked_output = model(data_with_mask, mask_idx=top_k_clusters)["output"]
                inverse_masked_output = model(data_with_inverse_mask, mask_idx=list(set(range(data.num_cluster.item())) - set(top_k_clusters)))["output"]
            else:
                masked_output = model(data_with_mask, mask_idx=None)["output"]
                inverse_masked_output = model(data_with_inverse_mask, mask_idx=None)["output"]

            # MASKED OUTPUT as INPUT -> ONLY IMPORANT NODES: at the end we stay only with the important nodes
            # INVERSE MASKED OUTPUT as INPUT -> ONLY UNIMPORTANT NODES: at the end we stay only with the unimportant nodes
            if task == "classification":
                original_output = original_output>=0
                masked_output = masked_output>=0
                inverse_masked_output = inverse_masked_output>=0
                pos_fidelity = ((inverse_masked_output == original_output).float()).abs().mean()
                neg_fidelity = ((masked_output == original_output).float()).abs().mean()
                metric_ = (data.y == original_output).float().mean().item()
                
            else:
                original_output = original_output.squeeze()
                masked_output = masked_output.squeeze()
                inverse_masked_output = inverse_masked_output.squeeze()
                pos_fidelity = torch.abs(inverse_masked_output - original_output)
                neg_fidelity = torch.abs(masked_output - original_output)
                metric_= abs(data.y - original_output).float().mean().item()
        else:
            raise NotImplementedError(f"Explainer type {loaded['args']['explainer_type']} is not implemented.")
        pos_fidelitys.append(pos_fidelity)
        neg_fidelitys.append(neg_fidelity)
        metrics_.append(metric_)
        # if loaded["args"]["explainer_type"]=="SEAL":
        #     torch.save(torch.tensor(numbers_to_mask), f"/shared/results/common/z1185722/gnn/ids/{loaded_args['data_set']}_{args.percentage}_{loaded_args['split']}_{int(args.absolute)}_{int(args.mask_contribution)}.pt")

    metrics = {
        "dataset": loaded_args["data_set"],
        "explainer_type": loaded["args"]["explainer_type"],
        "split": loaded_args["split"],
        "pos_fidelity": 1-torch.tensor(pos_fidelitys).mean().item() if task=="classification" else torch.tensor(pos_fidelitys).mean().item(),
        "neg_fidelity": 1-torch.tensor(neg_fidelitys).mean().item() if task=="classification" else torch.tensor(neg_fidelitys).mean().item(),
        "metric_calc":np.mean(np.array(metrics_)).item(),
        "percentage": args.percentage,
        "absolute": args.absolute,
        "mask_contribution": args.mask_contribution
    }
    print(f"Number of mean masked nodes: {np.mean(s)}")
    print(f"Percentage", args.percentage)
    print(f"Metric calculated: {metrics['metric_calc']}")
    print(f"Positive Fidelity: {metrics['pos_fidelity']}")
    print(f"Negative Fidelity: {metrics['neg_fidelity']}")

    
    if args.save_path is not None:
        file_exists = os.path.exists(args.save_path)
        with open(args.save_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
        

if __name__ == "__main__":
    main()
