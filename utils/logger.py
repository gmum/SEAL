from abc import ABC, abstractmethod
from pathlib import Path
import wandb
from PIL import Image
class LoggerBase(ABC):
    """
    Base class for logging utilities.
    """
    def __init__(self, log_dir: str|Path):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def log_metrics(self, metrics:dict, prefix:str):
        ...
    
    @abstractmethod
    def close(self):
        ...
        
class DummyLogger(LoggerBase):
    """
    Dummy logger that does nothing.

    Args:
        LoggerBase (ABC): Base class for logging utilities.
    """
    def __init__(self, log_dir: str|Path):
        super().__init__(log_dir)
    
    def log_metrics(self, metrics:dict, prefix:str):
        pass
    
    def log_config(self, config: dict):
        pass      

    def close(self):
        pass
    

class WandBLogger(LoggerBase):
    """
    Logger that uses Weights & Biases for logging metrics and artifacts.
    Args:
        LoggerBase (ABC): Base class for logging utilities.
    """
    def __init__(self, log_dir: str | Path, experiment_name: str, project_name: str, config: dict, **kwargs):
        """
        Initializes the WandBLogger.
        Args:
            log_dir (str | Path): Directory to save logs.
            experiment_name (str): Name of the experiment.
            project_name (str): Name of the project.
            config (dict): Configuration dictionary.
        """
        super().__init__(log_dir)
        self.run = wandb.init(
            name=experiment_name,
            project=project_name,
            config=config,
            dir=self.log_dir,
            **kwargs
        )

    def log_metrics(self, metrics: dict, prefix: str):
        """
        Log metrics to Weights & Biases.
        Args:
            metrics (dict): Dictionary of metrics to log.
            prefix (str): Prefix for the metrics.
        """
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
        self.run.log(metrics)


    def log_config(self, config: dict):
        """
        Log configuration to Weights & Biases.
        Args:
            config (dict): Configuration dictionary.
        """
        self.run.log(config)

    def log_grads(self, model, step):
        """
        Log gradients of the model parameters to Weights & Biases.
        Args:
            model (torch.nn.Module): The model to log gradients from.
            step (int): The current training step.
        """
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.run.log({f'grad_{name}': wandb.Histogram(
                    param.grad.cpu().numpy())}, step=step)

    def log_weights(self, model, step):
        """
        Log weights of the model parameters to Weights & Biases.
        Args:
            model (torch.nn.Module): The model to log weights from.
            step (int): The current training step.
        """
        for name, param in model.named_parameters():
            self.run.log({f'weight_{name}': wandb.Histogram(
                param.cpu().numpy())}, step=step)

    def log_artifact(self, artifact_name: str, artifact_path: str):
        """
        Log an artifact to Weights & Biases.
        Args:
            artifact_name (str): Name of the artifact.
            artifact_path (str): Path to the artifact file.
        """
        print("Logging artifact:", artifact_name)
        artifact = wandb.Artifact(artifact_name, type='image')
        artifact.add_file(artifact_path)
        self.run.log_artifact(artifact)

    def close(self):
        """
        Close the Weights & Biases run.
        This method should be called at the end of the training process to ensure all logs are flushed
        """
        self.run.finish()