import argparse
import datetime
import numpy as np
import time
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from torchmetrics import Accuracy, MeanSquaredError, MeanAbsoluteError, AUROC, F1Score
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset

from datasetss import dataset
from utils.utils import train_one_epoch, evaluate_one_epoch, MetricList, explain_model, test_one_epoch
import copy
from utils import logger, creator
from torch.utils.data import WeightedRandomSampler
from datasetss.dataset import SYNTHETIC_DATASET, REAL_DATASET, CLASSIFICATION_DATASET, REGRESSION_DATASET, STATS_DATASET

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
    parser.add_argument('--regularize', type=float, default=0.0, help='Regularization parameter for the model weigts SEAL')
    parser.add_argument('--regularize-contribution', type=float, default=0.0, help='Regularization parameter for the model weights SEAL')

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
    parser.add_argument('--data-set', default='rings-count', choices=['covid','sol', 'cyp', 'herg', 'qm9', 'rings-count', 'rings-max','X','P','B','indole','PAINS','herg_K'],
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
    
    parser.add_argument('--explain', action='store_true', help='Explain the model predictions')
    parser.add_argument('--split', type=int, default=0, help='Split number for cross-validation')
    parser.add_argument('--warmup-epochs', type=int, default=50, help='Number of warmup epochs')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    return parser

def write_to_csv(metrics_folds: list, args):
    """
    Write the metrics for each fold to a CSV file.

    Args:
        metrics_folds (list): A list of dictionaries containing metrics for each fold.
        args: The command-line arguments.
    """
    output_dir = Path(args.output_dir)
    if args.task == 'classification':
        f1_scores = [d['F1'] for d in metrics_folds]
        accuracies = [d['accuracy'] for d in metrics_folds]
        aurocs = [d['AUROC'] for d in metrics_folds]
    else:
        mse_scores = [d['rmse'] for d in metrics_folds]
        mae_scores = [d['mae'] for d in metrics_folds]
    # Calculate mean and std
    write_path = output_dir / f"analyze_p_{args.data_set}.txt"
    with open(write_path, 'a+') as f:
        f.write(f'{{"reg":{args.regularize}, "values":{metrics_folds} }}\n')
    
    print("Dataset:", args.data_set)
    print("Regularization:", args.regularize)

    if args.task == 'classification':
        print("F1:     mean =", np.mean(f1_scores), ", std =", np.std(f1_scores))
        print("Accuracy: mean =", np.mean(accuracies), ", std =", np.std(accuracies))
        print("AUROC:   mean =", np.mean(aurocs), ", std =", np.std(aurocs))
    else:
        print("RMSE:   mean =", np.mean(mse_scores), ", std =", np.std(mse_scores))
        print("MAE:    mean =", np.mean(mae_scores), ", std =", np.std(mae_scores))



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
    # print(dataset_train)
    
    dataset_train = pd.concat([dataset_train, dataset_val],axis=0, ignore_index=True)
    featurizer=dataset.GraphFeaturizer(y_column='Y')

    merged_set = featurizer(dataset_train, dataset_kwargs)
    # val_set = featurizer(dataset_val, dataset_kwargs)
    test_set = featurizer(dataset_test, dataset_kwargs)
    
    data_loader_test = DataLoader(
        test_set,
        batch_size=1,
        num_workers=args.num_workers,
        drop_last=False
    )
    # print(merged_set)
    
    model_kwargs = {
        "model": args.model,
        "hidden_features": args.hidden_dim,
        "input_features": merged_set[0].x.shape[1],
        "drop": args.drop,
        "num_layers": args.num_layers,
        "task": args.task,
        "number_of_clusters": args.num_clusters,
        "regularize": args.regularize,
        "regularize_contribution": args.regularize_contribution,
        "weight_decay": args.weight_decay,
    }
    
    config_log={
        "lr": args.lr,
        "epoch": args.epochs,
        "batch_size": args.batch_size,
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim,
        "drop": args.drop,
        "regularize": args.regularize,
    }
    

    test_metrics = metrics

    
    start_time = time.time()
    
    count=0
    metrics_folds=[]
    kfold=KFold(n_splits=10, shuffle=True, random_state=args.seed)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(merged_set)):
        train_metrics = metrics
        valid_metrics = metrics
        max_test_metric=0 if args.task == 'classification' else float('inf')

        logg = logger.DummyLogger(
            log_dir=args.output_dir)
        logg.log_config(config_log)

        print(f"Fold {fold+1}/{kfold.n_splits}")

        # train_set = merged_set[train_ids]
        # val_set = merged_set[val_ids]
        train_set = [merged_set[i] for i in train_ids]
        val_set = [merged_set[i] for i in val_ids]
        # print(train_set)
        if args.task == 'classification':
            labels = torch.cat([data.y for data in train_set]).long()
            # print(labels)
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
            sampler=sampler,
            shuffle=False
        )

        else:
            data_loader_train = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            shuffle=True   
        )

        
        # print(data_loader_train)
        data_loader_val = DataLoader(
            val_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False
        )
        
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

        print(f"Start training for {args.epochs} epochs on fold {fold}")
        min_val_loss= float('inf')
        best_state_dict = None
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
            # print(test_stats)
            if test_stats['loss'] < min_val_loss and epoch > args.warmup_epochs:
                min_val_loss = test_stats.get('loss')
                best_state_dict=copy.deepcopy(model.state_dict())
                count=0
            else:
                count+=1
            if count > args.patience and epoch > args.warmup_epochs:
                print("Early stopping")
                break

        
        model.load_state_dict(best_state_dict)
        model.eval()
        test_stats = evaluate_one_epoch(
                model, criterion, data_loader_val, device, logg, valid_metrics, args)
        metrics_folds.append(test_stats)
        print(test_stats)
        
        logg.close()
    print(metrics_folds)
    write_to_csv(metrics_folds, args)



    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'GNN training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)