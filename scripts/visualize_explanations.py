import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from utils.creator import create_model
from datasetss.dataset import SYNTHETIC_DATASET, STATS_DATASET, build_dataset, SyntheticGraphFeaturizer, GraphFeaturizer
import matplotlib.cm as cm
from torch_geometric.loader import DataLoader
import matplotlib.colors as mcolors

def load_explanation(data, idx):
    expl = data['explanations']
    return expl[idx]['node_mask'], expl[idx]['x'], expl[idx]['edge_index'], expl[idx]['y']

def visualize_atom_importance(smiles: str, atom_importance: np.ndarray, output_path: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    # normalize atom importance
    atom_importance = np.array(atom_importance)
    atom_importance = atom_importance / np.max(np.abs(atom_importance))

    # create colormap
    cmap = cm.get_cmap("RdBu_r", 20)
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1.0)
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    highlight_colors = [plt_colors.to_rgba(val) for val in atom_importance]
    atom_colors = {i: color[:3] for i, color in enumerate(highlight_colors)}
    # create drawer
    drawer = Draw.MolDraw2DSVG(800, 800)
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    opts.padding = 0.05
    opts.bondLineWidth = 2
    # draw molecule
    Draw.rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=list(atom_colors.keys()),
        highlightAtomColors=atom_colors
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # save SVG
    with open(output_path, 'w') as f:
        f.write(svg)

def main():
    parser = argparse.ArgumentParser(description="Visualize atom importance explanations.")
    parser.add_argument('--explanations_path', type=str, required=True, help='Path to explanations file')
    parser.add_argument('--output_dir', type=str, default='', help='Directory to save visualizations')
    parser.add_argument('--indices', type=int, nargs='+', required=True, help='Indices to visualize')
    args = parser.parse_args()
    
    loaded = torch.load(args.explanations_path, map_location=torch.device("cpu"), weights_only=False)
    loaded_args = loaded["args"]

    synthetic = loaded['model_args']['data_set'] in SYNTHETIC_DATASET
    mean=STATS_DATASET.get(loaded['model_args']['data_set'], {}).get("mean", 0.0)
    std=STATS_DATASET.get(loaded['model_args']['data_set'], {}).get("std", 1.0)

    dataset_kwargs = {
        "data_set": loaded['model_args']['data_set'],
        "mean": mean,
        "std": std,
        "y_column": loaded['model_args']['Y_column'],
        "smiles_col": "Drug",
        "split": loaded['model_args']['split']
    }


    _, _, dataset_test = build_dataset(dataset_kwargs)

    featurizer=GraphFeaturizer(y_column='Y') if not synthetic else  SyntheticGraphFeaturizer(y_column='Y')

    test_set = featurizer(dataset_test, dataset_kwargs)
    dataloader_test = DataLoader(test_set, batch_size=1, shuffle=False)



    model_kwargs = {
        "model": "SEAL",
        "hidden_features": loaded['model_args']['hidden_dim'],
        "num_layers": loaded['model_args']['num_layers'],
        "input_features": test_set[0].x.shape[1],
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_weights = torch.load(loaded["args"]["explainer_path"], map_location=torch.device(device))['state_dict']

    model = create_model(model_kwargs)
    model.load_state_dict(model_weights)
    model.eval()

    data = torch.load(args.explanations_path, map_location='cpu', weights_only=False)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for idx in args.indices:
        mask, x, edge_index, y = load_explanation(data, idx)
        smiles = dataset_test.iloc[idx].Drug
        output_path = f"{args.output_dir}/{loaded['model_args']['data_set']}_{idx}.svg"
        visualize_atom_importance(smiles, mask.cpu().numpy(), output_path=output_path)
        print(f"Saved visualization for index {idx} to {output_path}")

if __name__ == "__main__":
    main()
