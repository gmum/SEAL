import argparse
import datetime
import numpy as np
import time
import torch
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import  f1_score, mean_squared_error
from torchmetrics import Accuracy, MeanSquaredError, MeanAbsoluteError, AUROC, F1Score
from torch_geometric.loader import DataLoader
import csv
import os
from datasetss import dataset
from utils.utils import train_one_epoch, evaluate_one_epoch, MetricList, explain_model, test_one_epoch
import copy
from utils import logger, creator
from torch.utils.data import WeightedRandomSampler
from daatasetss.dataset import SYNTHETIC_DATASET, REAL_DATASET, CLASSIFICATION_DATASET, REGRESSION_DATASET, STATS_DATASET

def get_args_parser():
    parser = argparse.ArgumentParser(
        'GNN training script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)

    # Model parameters
    parser.add_argument('--model', default='SEAL', type=str, metavar='MODEL', choices=['SEAL'],
                        help='Name of model to train')
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--num-clusters', type=int, default=4)
    parser.add_argument('--regularize', type=float, default=0.0, help='Regularization parameter for the model weigts BRICSCONV')
    parser.add_argument('--regularize-contribution', type=float, default=0.0, help='Regularization parameter for the model weights BRICSCONV')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')

    # Optimizer parameters
    parser.add_argument('--optim', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',)

    # Learning rate schedule parameters
    parser.add_argument('--sched', default="", type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "None"')
    parser.add_argument("--step-size", type=int, default=30,
                        help="Step size for StepLR scheduler")
    parser.add_argument("--gamma", type=float, default=0.1,help="Gamma for StepLR scheduler") 
       

    # Dataset parameters
    parser.add_argument('--data-set', default='rings-count', choices=['covid','sol', 'cyp', 'herg', 'herg_K', 'rings-count', 'rings-max','X','P','B','indole','PAINS',],
                        type=str, help='dataset type')
    parser.add_argument('--task', default='classification',
                        type=str, choices=['regression', 'classification', 'multiclassification'],)
    parser.add_argument('--num-classes', type=int,
                        default=1, help='Number of classes')
    parser.add_argument('--Y_column', type=str,
                        default='Y', help='Column name for Y variable')

    # Other parameters  
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument("--split", default=0, type=int, help="Split index for dataset")
    parser.add_argument('--explain', action='store_true', help='Explain the model predictions')
    parser.add_argument('--save-path', default='default.pth', type=str, help='Path to save the model')
    parser.add_argument('--warmup-epochs', default=0, type=int, help='Number of warmup epochs for learning rate scheduler')
    parser.add_argument('--patience', default=30, type=int, help='Patience for early stopping')
    return parser


def write_to_csv(test_stats: dict,args):
    """
    Write the test statistics to a CSV file.

    Args:
        test_stats (dict): A dictionary containing test statistics.
        args (Namespace): The command-line arguments.
    """
    test_stats['dataset'] = args.data_set
    test_stats['task']= args.task
    test_stats['batch']=args.batch_size
    test_stats['dropout'] = args.drop
    test_stats['epoch']= args.epochs
    test_stats['hidden']= args.hidden_dim
    test_stats['lr']= args.lr
    test_stats['layers']= args.num_layers
    test_stats['model_type']= args.model
    test_stats['split'] = args.split

    csv_path =  f"{args.data_set}_SEAL_{args.regularize}.csv"
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=test_stats.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(test_stats)
        
    
    

def main(args):

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
    assert args.task in ['regression'] and args.data_set in REGRESSION_DATASET or args.task in ['classification'] and args.data_set in CLASSIFICATION_DATASET, "Task and dataset combination not supported"
    mean=STATS_DATASET.get(args.data_set, {}).get("mean", 0.0)
    std=STATS_DATASET.get(args.data_set, {}).get("std", 1.0)

    if args.task == 'regression':
        metrics = {
            "mae": MeanAbsoluteError(),
            "rmse": MeanSquaredError(squared=False),
        }

    elif args.task == 'classification':
        metrics={
            "accuracy": Accuracy(task='binary'),
            "AUROC": AUROC(task='binary'),
            "F1": F1Score(task='binary', average = "weighted"),
        }
    else:
        raise ValueError("Invalid task")
    dataset_kwargs = {
        "data_set": args.data_set,
        "mean": mean,
        "std": std,
        "y_column": args.Y_column,
        "smiles_col": "Drug",
        "split": args.split
    }

    metrics=MetricList(metrics)

    dataset_train, dataset_val, dataset_test = dataset.build_dataset(dataset_kwargs)
    featurizer=dataset.GraphFeaturizer(y_column='Y')

    train_set = featurizer(dataset_train, dataset_kwargs)
    val_set = featurizer(dataset_val, dataset_kwargs)
    test_set = featurizer(dataset_test, dataset_kwargs)

    if args.task == 'classification':
        labels = np.array([int(data.y.item() > 0.5) for data in train_set])

        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True  
        )
        data_loader_train = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            sampler=sampler
        )
    else:
        data_loader_train = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            shuffle=True
        )
    data_loader_val = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False
    )

    data_loader_test = DataLoader(
        test_set,
        batch_size=1,
        num_workers=args.num_workers,
        drop_last=False
    )

    model_kwargs = {
        "model": args.model,
        "hidden_features": args.hidden_dim,
        "input_features": train_set[0].x.shape[1],
        "drop": args.drop,
        "num_layers": args.num_layers,
        "task": args.task,
        "number_of_clusters": args.num_clusters,
        "regularize": args.regularize,
        "regularize_contribution": args.regularize_contribution,
        "weight_decay": args.weight_decay,
    }

    print(f"Creating model: {args.model}")
    model = creator.create_model(model_kwargs)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer_kwargs={
        "optim": args.optim,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,

    }

    optimizer = creator.create_optimizer(model, optimizer_kwargs)

    scheduler_kwargs = {
        "sched": args.sched,
        "step_size": args.step_size,
        "gamma": args.gamma,
    }

    lr_scheduler = creator.create_scheduler(optimizer, scheduler_kwargs)

    criterion = creator.create_loss(task=args.task)

    output_dir = Path(args.output_dir)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        # print(checkpoint)
        model.load_state_dict(checkpoint)
        if  'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            lr_scheduler.step(args.start_epoch)

    config_log={
        "lr": args.lr,
        "epoch": args.epochs,
        "batch_size": args.batch_size,
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim,
        "drop": args.drop,
        "regularize": args.regularize,
        "regularize_contribution": args.regularize_contribution,
    }

    logg = logger.WandBLogger(
        log_dir=args.output_dir, experiment_name=f'{args.model}_{args.data_set}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}', project_name='gnn_interpretability', config=config_log)

    model = model.to(device)
    logg.log_config(config_log)

    train_metrics = metrics
    valid_metrics = metrics
    test_metrics = metrics

    print(f"Start training for {args.epochs} epochs")

    start_time = time.time()
    best_state_dict=copy.deepcopy(model.state_dict())

    min_val_loss=float('inf')
    count=0
    for epoch in tqdm(range(args.start_epoch, args.epochs), total=args.epochs-args.start_epoch+1):

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, logg, train_metrics,
            args=args,
        )
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)
        test_stats = evaluate_one_epoch(
            model, criterion, data_loader_val, device, logg, valid_metrics, args)
        print(test_stats)
        if test_stats['loss'] < min_val_loss and epoch > args.warmup_epochs:
            min_val_loss = test_stats.get('loss')
            best_state_dict=copy.deepcopy(model.state_dict())
            count=0
        else:
            count+=1
        if count > args.patience and epoch > args.warmup_epochs:
            print("Early stopping")
            break

    print("Testing")
    print("Loading best model")
    model.load_state_dict(best_state_dict)
    model.eval()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    test_stats = evaluate_one_epoch(
            model, criterion, data_loader_val, device, logg, test_metrics, args)
    if args.task == 'classification':
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_args": model_kwargs,
                "args": vars(args),
                "f1": test_stats['F1'],
                "roc_auc": test_stats['AUROC'],
                "accuracy": test_stats['accuracy'],
            },
            args.save_path,
        )
    else:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_args": model_kwargs,
                "args": vars(args),
                "rmse": test_stats['rmse'],
                "mae": test_stats['mae'],
            },
            args.save_path,
        )

    test_stats['stop_epoch'] = epoch
    test_stats['time'] = total_time_str
    print(test_stats)
    
    write_to_csv(test_stats, args)
    print('Training time {}'.format(total_time_str))

    if args.explain:
        print("EXPLAINING")
        explain_model(indices=range(25), rows=5, cols=5,output_filename=f"{model_kwargs['model']}_importance_{dataset_kwargs['data_set']}.png",logger=logg ,model=model, dataset_test=dataset_test,test_set=test_set,model_kwargs=model_kwargs, args=args)

    logg.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'GNN training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)