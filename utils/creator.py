import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from architectures import SEALNetwork
import torch


def create_model(kwargs: dict) -> torch.nn.Module:
    """
    Create a model based on the provided configuration.

    Args:
        kwargs (dict): Configuration dictionary containing the model type and parameters.

    Returns:
        torch.nn.Module: An instance of the specified model class.
    """
    models = {
        "SEAL": SEALNetwork,
    }
    model_name = kwargs.get("model", "")
    if model_name not in models:
        raise ValueError(f"Unknown model name: {model_name}")


    return models[model_name](kwargs)


def create_optimizer(model: torch.nn.Module, kwargs: dict) -> torch.optim.Optimizer:
    """
    Create an optimizer for the given model based on the provided configuration.

    Args:
        model (torch.nn.Module): The model to optimize.
        kwargs (dict): Configuration dictionary containing optimizer parameters.

    Returns:
        torch.optim.Optimizer: An instance of the specified optimizer.
    """
    
    optimizers = {
        "sgd": optim.SGD,
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "rmsprop": optim.RMSprop
    }

    optim_name = kwargs.get("optim", "").lower()
    if optim_name not in optimizers:
        raise ValueError(f"Unknown optimizer name: {optim_name}")

    # Extract parameters and optimizer-specific arguments
    optimizer_class = optimizers[optim_name]
    lr = kwargs.get("lr", 0.001)  
    weight_decay = kwargs.get("weight_decay", 0.0) 
    momentum = kwargs.get("momentum", 0.9) if optim_name == "sgd" else None

    optimizer_params = {"lr": lr, "weight_decay": weight_decay}
    if momentum is not None:
        optimizer_params["momentum"] = momentum 

    return optimizer_class(model.parameters(), **optimizer_params)


def create_scheduler(optimizer: torch.optim.Optimizer, kwargs: dict) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create a learning rate scheduler for the given optimizer based on the provided configuration.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule.
        kwargs (dict): Configuration dictionary containing scheduler parameters.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: An instance of the specified learning rate scheduler.
    """

    sched_name = kwargs.get("sched", "").lower()
    if sched_name in ("", "none", "None"):
        return None  # No scheduler is used

    schedulers = {
        "step_lr": lr_scheduler.StepLR,
        "exponential_lr": lr_scheduler.ExponentialLR,
        "cosine_annealing_lr": lr_scheduler.CosineAnnealingLR,
        "reduce_lr_on_plateau": lr_scheduler.ReduceLROnPlateau
    }

    if sched_name not in schedulers:
        raise ValueError(f"Unknown scheduler name: {sched_name}")

    scheduler_class = schedulers[sched_name]

    # Setup scheduler-specific parameters
    if sched_name == "step_lr":
        step_size = kwargs.get("step_size", 10)
        gamma = kwargs.get("gamma", 0.1)
        scheduler = scheduler_class(
            optimizer, step_size=step_size, gamma=gamma)
    elif sched_name == "exponential_lr":
        gamma = kwargs.get("gamma", 0.95)
        scheduler = scheduler_class(optimizer, gamma=gamma)
    else:
        raise ValueError("Scheduler not implemented.")

    return scheduler

def create_loss(task: dict) -> torch.nn.Module:
    """
    Create a loss function based on the task type.

    Args:
        task (dict): A dictionary containing task information, including the task type.

    Raises:
        ValueError: If the task type is unknown.

    Returns:
        torch.nn.Module: An instance of the appropriate loss function.
    """
    if task == "classification":
        return torch.nn.BCEWithLogitsLoss()
    elif task == "regression":
        return torch.nn.MSELoss()
    else:
        raise ValueError(f"Unknown task: {task}")