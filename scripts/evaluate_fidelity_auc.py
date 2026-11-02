"""
evaluate_fidelity_auc.py

Computes a fragment-level fidelity AUC curve by iteratively masking or deleting
BRICS fragments (subgraphs) and recording a delta at each removal step.

Protocol:
    1. Rank all fragments by their importance score (descending = most important first).
    2. Remove fragments one at a time from the ranked list.
    3. At each step k (k fragments removed):
         Regression    : delta = |f(G_{-k}) - f(G_full)|          (continuous)
         Classification: delta = int(pred(G_{-k}) != pred(G_full)) (binary flip)
    4. Stop when only 1 fragment would remain (i.e., K-1 removal steps total).
    5. Compute AUC of the delta curve (normalized x-axis [0,1]).

    For regression the AUC captures how much removing important fragments shifts
    the predicted value.  For classification it captures the fraction of removal
    steps at which the model prediction flips — equivalent to the standard
    fidelity metric but resolved at fragment granularity.

Two masking modes:
    - "mask"  : zero out node features of removed fragments, keep graph topology.
    - "delete": fully remove nodes from the graph (updates edge_index, s, s_node, batch).

Fragment scores:
    - SEAL / SEALAtom : node_mask already encodes per-cluster scores (cluster score
      propagated to each node). We group nodes by hard cluster assignment and take
      the mean score per cluster (all nodes in a cluster share the same value anyway).
    - Other explainers: node-level attributions are aggregated per fragment using
      the BRICS assignment stored in data.s (same fragment structure used by SEAL).
      Aggregation method is configurable (mean | sum | max).
"""

import argparse
import copy
import csv
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
from torch_geometric.nn.conv import MessagePassing, GINConv
from torch_geometric.nn import Sequential, GCNConv, GATConv, GAT, GCN, GIN
from torch_geometric.nn import global_add_pool, global_max_pool
from torch.nn import Linear
from torch.nn import functional as F

# Path setup
sys.path.append("/home/musial/gnn/libs/ProtGNN")
sys.path.append("/home/musial/gnn")
sys.path.append("/home/musial/gnn/SEAL/architectures")
sys.path.append("/home/musial/gnn/SEAL/datasetss")
sys.path.append("/home/musial/gnn/SEAL")

from datasetss import GraphFeaturizer, build_dataset
# from trainGNNs import get_model, set_seed
from architectures import SEALNetwork, SEALAtom
from extract_explanations_mol import create_model

from libs.ProtGNN.Configures import model_args
from libs.ProtGNN.models import GnnNets
import random
import torch
import numpy as np

def get_model(model, input_features, num_classes, hidden_features, num_layers, linear_dim=32,drop=0.2):
    model_type = model
    num_node_features=input_features
    hidden_dim=hidden_features
    dropout=drop
    if model_type == "GCN":
        model = Sequential(
            "x, edge_index, batch",
            [
                (
                    GCN(
                        num_node_features,
                        hidden_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                        norm="batch_norm",
                    ),
                    "x, edge_index -> x",
                ),
                (global_add_pool, "x, batch -> x"),
                Linear(hidden_dim, num_classes),
                # torch.nn.Sigmoid(),
            ],
        )

    elif model_type == "GAT":
        model = Sequential(
            "x, edge_index, batch",
            [
                (
                    GAT(
                        num_node_features,
                        hidden_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                        norm="batch_norm",
                    ),
                    "x, edge_index -> x",
                ),
                (global_add_pool, "x, batch -> x"),
                Linear(hidden_dim, num_classes),
                # torch.nn.Sigmoid(),
            ],
        )
    elif model_type == "GIN":
        model = Sequential(
            "x, edge_index, batch",
            [
                (
                    GIN(
                        num_node_features,
                        hidden_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                        norm="batch_norm",
                    ),
                    "x, edge_index -> x",
                ),
                (global_add_pool, "x, batch -> x"),
                Linear(hidden_dim, num_classes),
                # torch.nn.Sigmoid(),
            ],
        )
    else:
        assert False
    print(f"Created model: {model_type} with {num_layers} layers. Numclasses: {num_classes}")
    return model
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def args_parser():
    parser = argparse.ArgumentParser(
        description="Fragment-level fidelity AUC evaluation."
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to append CSV results.")
    parser.add_argument("--explanations_path", type=str, required=True,
                        help="Path to saved explanations .pt file.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved model .pt file.")
    parser.add_argument(
        "--mode", type=str, default="delete",
        choices=["mask", "delete", "mask_contribution"],
        help=(
            "'mask'             : zero out node features of removed fragment nodes."
            "'delete'           : fully remove nodes from the graph (updates edge_index, s, batch)."
            "'mask_contribution': SEAL/SEALAtom only — pass mask_idx to the model's aggregation"
            "                     layer so removed cluster contributions are excluded from the sum."
            "                     Node features and graph topology are NOT modified."
        ),
    )
    parser.add_argument(
        "--aggregation", type=str, default="mean",
        choices=["mean", "sum", "max"],
        help="How to aggregate node-level scores into a per-fragment score "
             "(used for non-SEAL explainers).",
    )
    parser.add_argument(
        "--absolute", type=int, default=0,
        help=(
            "SEAL/SEALAtom only: 1 = use |node_mask| when ranking fragments."
            "SEAL cluster contributions can be negative, so this controls whether"
            "importance is defined by magnitude or signed value."
            "Has no effect for other explainers (saliency scores are always >= 0)."
        ),
    )
    return parser.parse_args()


# Fragment helpers

def get_assignment_matrix(data: Data, explainer_type: str) -> torch.Tensor:
    """Return the soft node-to-fragment assignment matrix [num_nodes, num_frags]."""
    if explainer_type == "SEALAtom":
        return data.s_node
    # SEAL and all other explainers share data.s (BRICS assignment)
    return data.s


def get_fragment_scores(
    node_mask: torch.Tensor,
    assignment: torch.Tensor,
    aggregation: str = "mean",
) -> torch.Tensor:
    """
    Aggregate node-level scores to per-fragment scores.

    Args:
        node_mask   : [num_nodes] importance score per node.
        assignment  : [num_nodes, num_frags] soft assignment.
        aggregation : 'mean' | 'sum' | 'max'.

    Returns:
        [num_frags] score per fragment.
    """
    frag_ids = assignment.argmax(dim=-1)          # [num_nodes] hard assignment
    num_frags = assignment.shape[1]
    device = node_mask.device

    scores = torch.zeros(num_frags, device=device)
    counts = torch.zeros(num_frags, device=device)

    for node_idx in range(len(frag_ids)):
        fid = frag_ids[node_idx].item()
        scores[fid] += node_mask[node_idx]
        counts[fid] += 1.0

    if aggregation == "mean":
        valid = counts > 0
        scores[valid] = scores[valid] / counts[valid]
    elif aggregation == "max":
        # Re-compute using scatter max
        scores = torch.full((num_frags,), float("-inf"), device=device)
        for node_idx in range(len(frag_ids)):
            fid = frag_ids[node_idx].item()
            scores[fid] = max(scores[fid].item(), node_mask[node_idx].item())
        scores[scores == float("-inf")] = 0.0
    # "sum" is already computed above

    return scores


def fragment_node_masks(assignment: torch.Tensor) -> List[Tuple[int, torch.Tensor]]:
    """
    Return list of (fragment_id, bool_mask) where bool_mask[i]=True means
    node i belongs to this fragment.  Only non-empty fragments are included.
    """
    frag_ids = assignment.argmax(dim=-1)   # [num_nodes]
    num_frags = assignment.shape[1]
    result = []
    for fid in range(num_frags):
        mask = frag_ids == fid
        if mask.any():
            result.append((fid, mask))
    return result



# Graph modification

def apply_fragment_removal(
    data: Data,
    remove_mask: torch.Tensor,
) -> Data:
    """
    Delete nodes from a graph given a boolean removal mask.
    Updates edge_index, batch, s, s_node, and edge-level mask tensors.

    Args:
        data        : PyG Data object (single graph, batch_size=1).
        remove_mask : [num_nodes] bool — True = remove this node from the graph.

    Returns:
        Modified deep copy of data with those nodes fully removed.
    """
    data_mod = copy.deepcopy(data)
    keep_mask = ~remove_mask
    keep_indices = keep_mask.nonzero(as_tuple=True)[0]

    if keep_indices.numel() == 0:
        # Degenerate: all nodes removed — return empty-feature graph
        data_mod.x = data_mod.x[keep_indices]
        data_mod.edge_index = torch.zeros((2, 0), dtype=torch.long,
                                          device=data.edge_index.device)
        if data.batch is not None:
            data_mod.batch = data.batch[keep_indices]
        return data_mod

    edge_index_new, _, _ = subgraph(
        keep_indices,
        data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
        return_edge_mask=True,
    )
    data_mod.x = data.x[keep_indices]
    data_mod.edge_index = edge_index_new

    if data.batch is not None:
        data_mod.batch = data.batch[keep_indices]

    # Update BRICS / cluster assignment matrices
    if hasattr(data, "s") and data.s is not None:
        data_mod.s = data.s[keep_indices]
    if hasattr(data, "s_node") and data.s_node is not None:
        data_mod.s_node = data.s_node[keep_indices]
    if hasattr(data, "mask") and data.mask is not None:
        data_mod.mask = data.mask[
            # mask is edge-level; filter to surviving edges
            subgraph(keep_indices, data.edge_index, relabel_nodes=False,
                     num_nodes=data.num_nodes, return_edge_mask=True)[2]
        ]
    if hasattr(data, "mask_atom") and data.mask_atom is not None:
        data_mod.mask_atom = data.mask_atom[
            subgraph(keep_indices, data.edge_index, relabel_nodes=False,
                     num_nodes=data.num_nodes, return_edge_mask=True)[2]
        ]

    return data_mod


# Unified model forward

@torch.no_grad()
def forward(model, data: Data, explainer_type: str,
            mask_idx=None, is_proto: bool = False) -> torch.Tensor:
    """
    Run the model and return the raw scalar output (logit or regression value).

    For SEAL/SEALAtom: passes the Data object directly with optional mask_idx.
    For standard GNNs: tries model(data) first; falls back to
                       model(x, edge_index, batch) if that raises TypeError.
    For ProtGNN      : output is (logits, ...) tuple; set is_proto=True to index [0].

    Args:
        mask_idx : SEAL/SEALAtom only — cluster indices to keep in the sum.
        is_proto : True when the model returns a (logits, ...) tuple (ProtGNN).
    """
    if explainer_type in ("SEAL", "SEALAtom", "SEALAuto"):
        return model(data, mask_idx=mask_idx)["output"].squeeze()
    else:
        try:
            out = model(data)
        except TypeError:
            out = model(data.x, data.edge_index, batch=data.batch)
        if is_proto:
            out = out[0]    # ProtGNN returns (logits, ...)
        return out.squeeze()


# Per-graph AUC curve

def compute_fidelity_curve(
    model,
    data: Data,
    node_mask: torch.Tensor,
    explainer_type: str,
    aggregation: str,
    mode: str,
    task: str,
    absolute: bool,
    is_proto: bool,
    device: torch.device,
) -> List[float]:
    """
    Compute the fidelity curve for a single graph.

    Removes fragments from most-important to least-important.
    Stops when only 1 fragment remains (K-1 removal steps).

    Delta per step:
        regression    : |f(G_{-k}) - f(G_full)|           (continuous)
        classification: float(pred(G_{-k}) != pred(G_full)) (0.0 or 1.0)

    Returns:
        List of delta values, one per removal step.
        Empty list if the graph has only 1 fragment.
    """
    assignment = get_assignment_matrix(data, explainer_type)         # [N, F]
    frag_list = fragment_node_masks(assignment)                       # [(fid, mask), ...]
    num_frags = len(frag_list)

    if num_frags <= 1:
        return []

    # Compute per-fragment scores
    node_mask_dev = node_mask.to(device)
    assignment_dev = assignment.to(device)
    frag_scores = get_fragment_scores(node_mask_dev, assignment_dev, aggregation)

    #  Full graph prediction (needed before sorting for SEAL sign logic)
    full_output = forward(model, data, explainer_type, is_proto=is_proto)
    if task == "classification":
        # ProtGNN outputs 2-class logits [class0, class1] → use argmax
        # Standard binary models output a scalar logit → use >= 0
        if is_proto:
            full_pred = full_output.argmax().float()        # scalar: 0 or 1
        else:
            full_pred = (full_output >= 0).float()   # scalar: 0 or 1

    # Sort fragments by importance
    # SEAL contributions can be negative.  Which direction is "important" depends
    # on the sign of the prediction:
    #   full_output >= 0 : positive contributions pushed the output up → mask highest first
    #   full_output <  0 : negative contributions pushed the output down → mask lowest first
    #   --absolute       : rank by magnitude regardless of sign → mask largest |score| first
    #
    # Other explainers (saliency etc.) are always non-negative → always descending.
    is_seal = explainer_type in ("SEAL", "SEALAtom", "SEALAuto")

    if is_seal and not absolute:
        # Sign-aware: most important = same sign as prediction, largest magnitude
        sort_descending = full_output.item() >= 0
    else:
        # absolute=True (magnitude) or non-SEAL (always non-negative) → largest first
        sort_descending = True

    sorted_frags = sorted(
        frag_list,
        key=lambda x: frag_scores[x[0]].item(),
        reverse=sort_descending,
    )

    # Iterative removal

    if mode == "mask_contribution" and not is_seal:
        raise ValueError(
            "'mask_contribution' mode is only supported for SEAL/SEALAtom models."
        )

    # Total number of clusters in the assignment matrix
    num_clusters = assignment.shape[1]

    deltas: List[float] = []
    cumulative_remove   = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    removed_cluster_ids: set = set()

    # Remove all but the last fragment (sorted_frags[-1] stays in the graph)
    for fid, frag_mask in sorted_frags[:-1]:
        removed_cluster_ids.add(fid)
        cumulative_remove |= frag_mask.to(device)

        try:
            if mode == "mask_contribution":
                # SEAL only
                # Combined intervention:
                #   1. Zero node features of removed fragments (input-level)
                #   2. Exclude their cluster contributions via mask_idx (aggregation-level)
                data_mod = copy.deepcopy(data)
                keep_mask = (~cumulative_remove).float().unsqueeze(-1)  # [N, 1]
                data_mod.x = data_mod.x * keep_mask
                remaining_clusters = sorted(
                    set(range(num_clusters)) - removed_cluster_ids
                )
                out = forward(model, data_mod, explainer_type, mask_idx=remaining_clusters, is_proto=is_proto)

            elif mode == "mask":
                # Zero node features    
                # Keep graph topology; set removed nodes' features to 0.
                data_mod = copy.deepcopy(data)
                keep_mask = (~cumulative_remove).float().unsqueeze(-1)  # [N, 1]
                data_mod.x = data_mod.x * keep_mask
                out = forward(model, data_mod, explainer_type, is_proto=is_proto)

            else:  # "delete"
                # Remove nodes from graph
                # Fully deletes nodes: updates edge_index, s, s_node, batch.
                data_mod = apply_fragment_removal(data, cumulative_remove)
                out = forward(model, data_mod, explainer_type, is_proto=is_proto)

        except Exception as exc:
            print(f"  Warning [frag {fid}, mode={mode}]: {exc}")
            deltas.append(float("nan"))
            continue

        if task == "regression":
            delta = torch.abs(out - full_output).item()
        else:  # classification
            # ProtGNN: argmax over 2-class logits; standard binary: >= 0 threshold
            if is_proto:
                mod_pred = out.argmax().float()
            else:
                mod_pred = (out >= 0).float()
            delta = float(mod_pred != full_pred)   # 1.0 if flipped, else 0.0

        deltas.append(delta)

    return deltas


def auc_of_curve(deltas: List[float]) -> float:
    """
    Normalized trapezoidal AUC over a fidelity curve (x-axis in [0, 1]).

    Regression    : AUC of |f(G_{-k}) - f(G_full)| — higher means the removed
                    fragments mattered more (good explanation = rises quickly).
    Classification: AUC of prediction-flip indicator (0/1) — fraction of the
                    removal steps where the prediction changed.  Equivalent to
                    standard fidelity+ but averaged across fragment granularities.

    NaN steps are dropped; surviving points keep their relative x positions.
    """
    clean = [(i, d) for i, d in enumerate(deltas) if not np.isnan(d)]
    if len(clean) < 2:
        return float("nan")

    xs = np.array([c[0] for c in clean], dtype=float)
    ys = np.array([c[1] for c in clean], dtype=float)
    xs_norm = xs / (len(deltas) - 1)          # normalize to [0, 1]
    return float(np.trapz(ys, xs_norm))


# Model loading

def build_dataset_from_args(loaded_args):
    """Construct dataset kwargs from loaded model args."""
    ds = loaded_args["data_set"]
    stats = {
        "sol":   {"mean": -2.86, "std": 2.38},
        "sol_s": {"mean": -2.51, "std": 2.20},
    }
    mean = stats[ds]["mean"] if ds in stats else 0.0
    std  = stats[ds]["std"]  if ds in stats else 1.0
    return {
        "data_set":   ds,
        "mean":       mean,
        "std":        std,
        "y_column":   "Y",
        "smiles_col": "Drug",
        "task":       loaded_args["task"],
        "split":      loaded_args.get("split", 0),
    }


def load_model(loaded_model, test_set, device):
    """
    Load the GNN model for inference (no Captum wrapper needed).

    Returns:
        model    : raw torch.nn.Module in eval mode
        is_proto : True if ProtGNN model (argmax classification output)
    """
    loaded_args = loaded_model["args"]
    is_proto = False
    try:
        model = get_model(
            model=loaded_args["model"],
            input_features=test_set[0].x.shape[1],
            num_classes=1,
            hidden_features=loaded_args["hidden_dim"],
            num_layers=loaded_args["num_layers"],
        ).to(device)
        print(loaded_args)
        model.load_state_dict(loaded_model["state_dict"])
        for module in model.modules():
            if isinstance(module, MessagePassing) and not hasattr(module, "in_channels"):
                ch = module.nn.channel_list
                module.in_channels, module.out_channels = ch[0], ch[-1]
        # No wrapper needed: forward() handles both calling conventions directly
    except Exception:
        model_args.model_name = loaded_model["args"]["model_type"]
        model_args.readout = loaded_model["args"]["readout"]
        model_args.num_prototypes_per_class = 3
        model_args.latent_dim = (
            [loaded_model["args"]["hidden"]] * loaded_model["args"]["num_layers"]
        )
        model_args.latent_dim[-1] = 128
        print(model_args)
        protgnn = GnnNets(11, 2, model_args)
        for layer in protgnn.model.gnn_layers:
            if isinstance(layer, GINConv):
                layer.in_channels = layer.nn[0].in_features
                layer.out_channels = layer.nn[0].out_features
        protgnn.load_state_dict(loaded_model["state_dict"])
        model = protgnn.model   # use inner model directly; forward() handles output indexing
        is_proto = True
    model = model.to(device)
    model.eval()
    return model, is_proto


def load_seal_model(loaded, test_set, loaded_args, device):
    """Load a SEAL / SEALAtom model."""
    explainer_type = loaded["args"]["explainer_type"]
    model_weights = torch.load(
        loaded["args"]["model_path"],
        map_location=device,
        weights_only=False,
    )["state_dict"]
    model_kwargs = {
        "model":             explainer_type,
        "hidden_features":   loaded_args["hidden_dim"],
        "input_features":    test_set[0].x.shape[1],
        "drop":              0.0,
        "num_layers":        loaded_args["num_layers"],
        "task":              loaded_args["task"],
        "number_of_clusters": loaded.get("explainer", {}).get(
            "number_of_clusters", loaded_args.get("num_clusters", 5)
        ),
        "regularize":        0,
        "weight_decay":      0,
    }
    print(model_kwargs)
    model = create_model(model_kwargs)
    model.load_state_dict(model_weights)
    model = model.to(device)
    model.eval()
    return model


# Main

def main():
    args = args_parser()
    print(args)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoints   
    loaded_model = torch.load(args.model_path, map_location="cpu", weights_only=False)
    loaded       = torch.load(args.explanations_path, weights_only=False)

    loaded_args    = loaded_model["args"]
    task           = loaded_args["task"]
    explainer_type = loaded["args"]["explainer_type"]

    # Build dataset
    dataset_kwargs = build_dataset_from_args(loaded_args)
    _, _, dataset_test = build_dataset(dataset_kwargs)
    featurizer  = GraphFeaturizer(y_column="Y")
    test_set    = featurizer(dataset_test, dataset_kwargs)
    dataloader  = DataLoader(test_set, batch_size=1, shuffle=False)

    # Load model
    is_proto = False
    if explainer_type in ("SEAL", "SEALAtom", "SEALAuto"):
        model = load_seal_model(loaded, test_set, loaded_args, device)
    else:
        model, is_proto = load_model(loaded_model, test_set, device)

    explanations = loaded["explanations"]

    # Evaluation loop
    auc_scores   = []
    skipped      = 0

    for ids, (e, data) in tqdm(
        enumerate(zip(explanations, dataloader)),
        total=len(explanations),
        desc="Evaluating",
    ):
        assert torch.equal(e.x, data.x), \
            f"Graph {ids}: explanation and data node features do not match!"
        data = data.to(device)
        if explainer_type == "SEALAuto":
            assignment = e.s if hasattr(e, "s") else None
            if assignment is None:
                with torch.no_grad():
                    assignment = model(data, mask_idx=None)["s"]
            if assignment.dim() == 3 and assignment.size(0) == 1:
                assignment = assignment[0]
            data.s = assignment.to(device)

        # Extract node-level importance scores
        node_mask = e["node_mask"].to(device)           # [num_nodes]

        # Absolute ranking is SEAL/SEALAtom only — cluster contributions can be
        # negative, so |mask| ranks by magnitude rather than signed value.
        # Saliency-based explainers always produce non-negative scores.
        if explainer_type in ("SEAL", "SEALAtom", "SEALAuto") and args.absolute:
            node_mask = torch.abs(node_mask)

        # Compute fidelity curve
        curve = compute_fidelity_curve(
            model=model,
            data=data,
            node_mask=node_mask,
            explainer_type=explainer_type,
            aggregation=args.aggregation,
            mode=args.mode,
            task=task,
            absolute=bool(args.absolute),
            is_proto=is_proto,
            device=device,
        )

        if len(curve) == 0:
            # Only 1 fragment — cannot compute curve
            skipped += 1
            continue

        auc = auc_of_curve(curve)
        if not np.isnan(auc):
            auc_scores.append(auc)

    # Aggregate results
    print(f"Skipped (1-fragment graphs): {skipped}")
    print(f"Valid graphs evaluated: {len(auc_scores)}")

    mean_auc = float(np.mean(auc_scores)) if auc_scores else float("nan")
    std_auc  = float(np.std(auc_scores))  if auc_scores else float("nan")

    print(f"Mean AUC fidelity : {mean_auc:.6f}")
    print(f"Std  AUC fidelity : {std_auc:.6f}")

    metrics = {
        "dataset":        loaded_args["data_set"],
        "explainer_type": explainer_type,
        "split":          loaded_args.get("split", 0),
        "task":           task,
        "mode":           args.mode,
        "aggregation":    args.aggregation,
        "absolute":       args.absolute,
        "mean_auc":       mean_auc,
        "std_auc":        std_auc,
        "n_valid":        len(auc_scores),
        "n_skipped":      skipped,
    }

    if args.save_path is not None:
        file_exists = os.path.exists(args.save_path)
        with open(args.save_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
        print(f"Results appended to: {args.save_path}")


if __name__ == "__main__":
    main()
