
import xml.etree.ElementTree as ET
from matplotlib.gridspec import GridSpec
import cairosvg
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import Dict
import copy
import io
from rdkit import Chem
import matplotlib
import matplotlib.cm as cm
import numpy as np
from rdkit.Chem.Draw import rdMolDraw2D
import pandas as pd
from utils.logger import LoggerBase

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, logger: LoggerBase, metrics, args) -> dict:
    """
    Train the model for one epoch.
    Args:
        model (torch.nn.Module): The model to train.
        criterion (torch.nn.Module): The loss function.
        data_loader (torch.utils.data.DataLoader): The data loader for training data.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        device (torch.device): The device to train the model on.
        epoch (int): The current epoch number.
        logger (LoggerBase): The logger for logging metrics.
        metrics: The metrics tracker for tracking training metrics.
        args (Namespace): The command-line arguments.

    Returns:
        dict: The training metrics.
    """

    model.train()
    model= model.to(device)
    for i, batch in enumerate(data_loader):
        # print(batch)
        batch = batch.to(device)

        output = model(batch)
        out = output["output"]
        loss = criterion(out, batch.y.reshape(-1, 1)) + output["losses"]

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        metrics.update(out, batch.y.reshape(-1, 1))

        train_metrics = {"loss": loss.item(), **metrics.compute()}

        logger.log_metrics(train_metrics, prefix="train")

    return train_metrics


def evaluate_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device, logger: LoggerBase, metrics, args) -> dict:
    """
    Evaluate the model for one epoch.
    Args:
        model (torch.nn.Module): The model to evaluate.
        criterion (torch.nn.Module): The loss function.
        data_loader (torch.utils.data.DataLoader): The data loader for evaluation data.
        device (torch.device): The device to evaluate the model on.
        logger (LoggerBase): The logger for logging metrics.
        metrics: The metrics tracker for tracking evaluation metrics.
        args (Namespace): The command-line arguments.
    Returns:
        dict: The evaluation metrics.
    """
    
    model.eval()
    model= model.to(device)

    losses=[]
    with torch.no_grad():
        for i, batch in enumerate(data_loader):

            batch = batch.to(device)

            output = model(batch)
            out = output["output"]
            loss = criterion(out, batch.y.reshape(-1, 1)) + output["losses"]

            losses.append(loss.item())
            metrics.update(out, batch.y.reshape(-1, 1))

        val_metrics = {"loss": np.mean(losses).item(), **metrics.compute()}

        logger.log_metrics(val_metrics, prefix="val")

    return val_metrics

def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device, logger: LoggerBase, metrics, args) -> dict:
    """
    Test the model for one epoch.
    Args:
        model (torch.nn.Module): The model to test.
        criterion (torch.nn.Module): The loss function.
        data_loader (torch.utils.data.DataLoader): The data loader for test data.
        device (torch.device): The device to test the model on.
        logger (LoggerBase): The logger for logging metrics.
        metrics : The metrics tracker for tracking test metrics.
        args (Namespace): The command-line arguments.
    Returns:
        dict: The test metrics.
    """

    model.eval()
    model= model.to(device)
    losses=[]
    with torch.no_grad():
        for i, batch in enumerate(data_loader):

            batch = batch.to(device)

            output = model(batch)
            out = output["output"]
            loss = criterion(out, batch.y.reshape(-1, 1)) + output["losses"]

            losses.append(loss.item())
            metrics.update(out, batch.y.reshape(-1, 1))

        val_metrics = {"loss": np.mean(losses).item(), **metrics.compute()}

        logger.log_metrics(val_metrics, prefix="test")

    return val_metrics


def explain_model(indices: list, rows: int, cols: int, output_filename: str, logger: LoggerBase, model: torch.nn.Module, dataset_test: pd.DataFrame, test_set: torch.utils.data.Dataset, model_kwargs: dict, args):
    # Set up the figure
    fig = plt.figure(figsize=(cols*6, rows*6))
    gs = GridSpec(rows, cols, figure=fig)

    # Set up the grid for subplots
    max_molecules = min(len(indices), rows * cols)

    # Iterate through the indices and create subplots
    for i, idx in enumerate(indices[:max_molecules]):
        try:
            if i >= rows * cols:
                break
            # Calculate the row and column for the subplot
            row = i // cols
            col = i % cols

            # Create the SVG string for the molecule, 
            svg_str = visualize_multiple(model, dataset_test, test_set, idx,model_kwargs)
            png_data = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'))

            # Convert the SVG to a PNG image
            img = Image.open(io.BytesIO(png_data))

            img_array = np.array(img)

            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img_array)
            ax.set_title(f"Molecule {idx}")
            ax.axis('off')
        except Exception as e:
            print(f"Error processing molecule {idx}: {e}")

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Combined image saved as {output_filename}")
    logger.log_artifact("Importance", output_filename)
    
    
    return             


    

def visualize_multiple(model: torch.nn.Module, dataset_test: pd.DataFrame, test_set: torch.utils.data.Dataset, idx: int, model_kwargs: dict, number_of_outputs=1):

    svgs = [visualize_importance(model, dataset_test, test_set, idx, model_kwargs, i) for i in range(number_of_outputs)]

    roots = [ET.fromstring(svg) for svg in svgs]

    new_svg = ET.Element('svg', {
        'width': str(len(roots) * 1500),
        'height': str(1500),
        'xmlns': 'http://www.w3.org/2000/svg'
    })
    
    gs= [ET.SubElement(new_svg,'g', {'transform': f'translate({1500*id_gs},0)'}) for id_gs in range(len(roots))]
    
    for g,root in zip(gs,roots):
        for child in root:
            g.append(child)
    
    new_svg_str = ET.tostring(new_svg, encoding='unicode')
    return new_svg_str


def visualize_importance(model: torch.nn.Module, dataset_test: pd.DataFrame, test_set: torch.utils.data.Dataset, idx: int, model_kwargs: dict, id_output: int):
    model = model.eval().to('cpu')
    mol = Chem.MolFromSmiles(dataset_test.iloc[idx].Drug)
    data = test_set[idx].to('cpu')

    out = model(data)
    cluster_indices = torch.argmax(out['s'], dim=-1)

    transformed_values = out['x_cluster_transformed'][id_output].flatten()
    # print(transformed_values)
    highlight_atoms = list(range(data.x.shape[0]))
    highlight_atoms = [int(atom) for atom in highlight_atoms]

    atom_values = []
    atom_without_transform = []

    for i, cluster_idx in enumerate(cluster_indices.tolist()):
        cluster_value = transformed_values[cluster_idx].item()
        atom_values.append(float(cluster_value))
        atom_without_transform.append(cluster_value)

    cmap = cm.get_cmap("RdBu_r", 20)
    min_=min(atom_values)
    max_=max(atom_values)
    a=np.array(atom_values)
    a=2*a/max(abs(a))

    atom_values = a.tolist()
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)

    highlight_colors = [plt_colors.to_rgba(val) for val in atom_values]
    highlight_kwargs = {
        'highlightAtoms': highlight_atoms,
        'highlightAtomColors': {int(atom): highlight_colors[i] for i, atom in enumerate(highlight_atoms)}
    }

    y_value = float(data.y.item()) if isinstance(
        data.y, torch.Tensor) else float(data.y)

    d = rdMolDraw2D.MolDraw2DSVG(1500, 1500)
    d.standardColoursForHighlightedAtoms = False  # Use standard colors for highlighted atoms

    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, **highlight_kwargs)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    svg = svg.replace('svg:', '')

    annotation = f'<text x="40" y="40" font-size="40" fill="black">y: {y_value:.3f}</text>'
    if model_kwargs['task'] == 'classification':
        annotation_pred = f'<text x="40" y="80" font-size="40" fill="black">y_pred: {torch.nn.functional.sigmoid(out["output"][-1]).item()>0.5:.3f} y_prob:{torch.nn.functional.sigmoid(out["output"][-1]).item():.3f}</text>'
    else:
        annotation_pred = f'<text x="40" y="80" font-size="40" fill="black">y_pred: {out["output"][-1].item():.3f}</text>'
    atom_texts = []

    for i, cluster_idx in enumerate(cluster_indices.unique()):
        atom_idx = (cluster_indices == cluster_idx).nonzero(as_tuple=False)
        atom_idx = atom_idx.flatten().numpy()[0]
        x, y = d.GetDrawCoords(int(atom_idx)) 
        cluster_value = atom_without_transform[int(atom_idx)]
        text_element = f'<text x="{x}" y="{y+55}" font-size="30" fill="black">{cluster_value:.2f}</text>'
        atom_texts.append(text_element)

    svg = svg.replace('</svg>', annotation + '</svg>')
    svg = svg.replace('</svg>', annotation_pred + '</svg>')
    svg = svg.replace('</svg>', ''.join(atom_texts) + '</svg>')

    return svg

class MetricList:
    def __init__(self, metrics:dict):
        """
        Initialize the MetricList with a dictionary of metrics.

        Args:
            metrics (dict): A dictionary of metric names and their corresponding metric functions.
        """
        self.metrics = copy.deepcopy(metrics)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update the metrics with the predictions and targets.

        Args:
            preds (torch.Tensor): The predicted values.
            targets (torch.Tensor): The ground truth values.
        """
        for name, metric in self.metrics.items():
            metric.update(preds.detach().cpu(), targets.cpu())

    def compute(self) -> Dict[str, float]:
        """
        Compute the current values of all metrics.

        Returns:
            Dict[str, float]: A dictionary containing the computed metric values.
        """
        metrics = {}
        for name, metric_fn in self.metrics.items():
            metrics[name] = metric_fn.compute().item()
            metric_fn.reset()
        return metrics
