import os
import sage
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import dynamic_selection as ds
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchmetrics import Accuracy
from captum.attr import DeepLift, IntegratedGradients
from dynamic_selection import BaseModel, MaskingPretrainer, GreedyDynamicSelection
from dynamic_selection.utils import Flatten, StaticMaskLayer1d

import sys
sys.path.append('../')
from baselines import DifferentiableSelector, ConcreteMask


# Set up command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='greedy',
                    choices=['sage', 'permutation', 'deeplift', 'intgrad',
                             'cae', 'greedy'])
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--num_trials', type=int, default=1)
parser.add_argument('--num_restarts', type=int, default=1)


# Various configurations.
num_features = list(range(5, 35, 5)) + list(range(40, 110, 10))
max_features = 50


# Helper function for network architecture.
def get_network(d_in, d_out):
    hidden = 512
    dropout = 0.3
    model = nn.Sequential(
        nn.Linear(d_in, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, d_out))
    return model


if __name__ == '__main__':
    # Parse args.
    args = parser.parse_args()
    device = torch.device('cuda', args.gpu)
    
    # Load and split dataset.
    mnist_dataset = MNIST('/tmp/mnist/', download=True, train=True,
                          transform=transforms.Compose([transforms.ToTensor(), Flatten()]))
    test_dataset = MNIST('/tmp/mnist/', download=True, train=False,
                         transform=transforms.Compose([transforms.ToTensor(), Flatten()]))
    np.random.seed(0)
    val_inds = np.sort(np.random.choice(len(mnist_dataset), size=10000, replace=False))
    train_inds = np.setdiff1d(np.arange(len(mnist_dataset)), val_inds)
    train_dataset = torch.utils.data.Subset(mnist_dataset, train_inds)
    val_dataset = torch.utils.data.Subset(mnist_dataset, val_inds)
    d_in = 784
    d_out = 10
    
    # Make results directory.
    if not os.path.exists('results'):
        os.makedirs('results')
    
    for trial in range(args.num_trials):
        # For saving results.
        results_dict = {
            'acc': {},
            'features': {}
        }
        acc_metric = Accuracy(task='multiclass', num_classes=10)
        
        if args.method in ['sage', 'permutation', 'deeplift', 'intgrad']:
            # Train model.
            model = get_network(d_in, d_out)
            basemodel = BaseModel(model).to(device)
            basemodel.fit(
                train_dataset,
                val_dataset,
                mbsize=128,
                lr=1e-3,
                nepochs=250,
                loss_fn=nn.CrossEntropyLoss(),
                verbose=False)

            # Calculate feature importance scores.
            if args.method == 'sage':
                model_activation = nn.Sequential(model, nn.Softmax(dim=1))
                default_values = ds.data.get_xy(train_dataset)[0].mean(dim=0).numpy()
                imputer = sage.DefaultImputer(model_activation, default_values)
                estimator = sage.PermutationEstimator(imputer, 'cross entropy')
                val_x, val_y = ds.data.get_xy(val_dataset)
                sage_values = estimator(val_x.numpy(), val_y.numpy(), thresh=0.01)
                ranked_features = np.argsort(sage_values.values)[::-1]

            elif args.method == 'permutation':
                permutation_importance = np.zeros(d_in)
                x_train = ds.data.get_xy(train_dataset)[0]
                for i in tqdm(range(d_in)):
                    val_x, val_y = ds.data.get_xy(val_dataset)
                    val_x[:, i] = x_train[np.random.choice(len(x_train), size=len(val_x)), i]
                    with torch.no_grad():
                        pred = model(val_x.to(device)).softmax(dim=1).cpu()
                    permutation_importance[i] = - acc_metric(pred, val_y)
                ranked_features = np.argsort(permutation_importance)[::-1]
                
            elif args.method == 'deeplift':
                deeplift = DeepLift(model, multiply_by_inputs=False)
                x, y = ds.data.get_xy(val_dataset)
                x = x.to(device)
                x.requires_grad = True
                y = y.to(device)
                attr = deeplift.attribute(x, target=y)
                mean_abs = np.abs(attr.cpu().data.numpy()).mean(axis=0)
                ranked_features = np.argsort(mean_abs)[::-1]
                
            elif args.method == 'intgrad':
                ig = IntegratedGradients(model)
                x, y = ds.data.get_xy(val_dataset)
                x = x.to(device)
                x.requires_grad = True
                y = y.to(device)
                baseline = x.mean(dim=0, keepdim=True).detach()
                attr = ig.attribute(x, baselines=baseline, target=y, internal_batch_size=len(x))
                mean_abs = np.abs(attr.cpu().data.numpy()).mean(axis=0)
                ranked_features = np.argsort(mean_abs)[::-1]
            
            # Train models with top features.
            for num in num_features:
                # Prepare module to mask all but top features
                selected_features = ranked_features[:num]
                inds = torch.tensor(np.isin(np.arange(d_in), selected_features), device=device)
                mask_layer = StaticMaskLayer1d(inds)
            
                best_loss = np.inf
                for _ in range(args.num_restarts):
                    # Train model.
                    model = nn.Sequential(mask_layer, get_network(num, d_out))
                    basemodel = BaseModel(model).to(device)
                    basemodel.fit(
                        train_dataset,
                        val_dataset,
                        mbsize=128,
                        lr=1e-3,
                        nepochs=250,
                        loss_fn=nn.CrossEntropyLoss(),
                        verbose=False)
                    
                    # Check if best.
                    val_loss = basemodel.evaluate(val_dataset, nn.CrossEntropyLoss(), 1024)
                    if val_loss < best_loss:
                        best_model = basemodel
                        best_loss = val_loss

                # Evaluate using best model.
                acc = best_model.evaluate(test_dataset, acc_metric, 1024)
                results_dict['acc'][num] = acc
                results_dict['features'][num] = selected_features
                print(f'Num = {num}, Acc = {100*acc:.2f}')

        elif args.method == 'cae':
            for num in num_features:
                # Train model with differentiable feature selection.
                model = get_network(d_in, d_out)
                selector_layer = ConcreteMask(d_in, num)
                diff_selector = DifferentiableSelector(model, selector_layer).to(device)
                diff_selector.fit(
                    train_dataset,
                    val_dataset,
                    mbsize=128,
                    lr=1e-3,
                    nepochs=250,
                    loss_fn=nn.CrossEntropyLoss(),
                    patience=5,
                    verbose=False)

                # Extract top features.
                logits = selector_layer.logits.cpu().data.numpy()
                selected_features = np.sort(logits.argmax(axis=1))
                if len(np.unique(selected_features)) != num:
                    print(f'{len(np.unique(selected_features))} selected instead of {num}, appending extras')
                    num_extras = num - len(np.unique(selected_features))
                    remaining_features = np.setdiff1d(np.arange(d_in), selected_features)
                    selected_features = np.sort(np.concatenate([np.unique(selected_features), remaining_features[:num_extras]]))

                # Prepare module to mask all but top features
                inds = torch.tensor(np.isin(np.arange(d_in), selected_features), device=device)
                mask_layer = StaticMaskLayer1d(inds)

                best_loss = np.inf
                for _ in range(args.num_restarts):
                    # Train model.
                    model = nn.Sequential(mask_layer, get_network(num, d_out))
                    basemodel = BaseModel(model).to(device)
                    basemodel.fit(
                        train_dataset,
                        val_dataset,
                        mbsize=128,
                        lr=1e-3,
                        nepochs=250,
                        loss_fn=nn.CrossEntropyLoss(),
                        verbose=False)

                    # Check if best.
                    val_loss = basemodel.evaluate(val_dataset, nn.CrossEntropyLoss(), 1024)
                    if val_loss < best_loss:
                        best_model = basemodel
                        best_loss = val_loss

                # Evaluate using best model.
                acc = best_model.evaluate(test_dataset, acc_metric, 1024)
                results_dict['acc'][num] = acc
                results_dict['features'][num] = selected_features
                print(f'Num = {num}, Acc = {100*acc:.2f}')

        elif args.method == 'greedy':
            # Prepare networks.
            predictor = get_network(d_in * 2, d_out)
            selector = get_network(d_in * 2, d_in)

            # Pretrain predictor
            mask_layer = ds.utils.MaskLayer(append=True)
            pretrain = MaskingPretrainer(predictor, mask_layer).to(device)
            pretrain.fit(
                train_dataset,
                val_dataset,
                mbsize=128,
                lr=1e-3,
                nepochs=100,
                loss_fn=nn.CrossEntropyLoss(),
                patience=5,
                verbose=False)

            # Train selector and predictor jointly.
            gdfs = GreedyDynamicSelection(selector, predictor, mask_layer).to(device)
            gdfs.fit(
                train_dataset,
                val_dataset,
                mbsize=128,
                lr=1e-3,
                nepochs=250,
                max_features=max_features,
                loss_fn=nn.CrossEntropyLoss(),
                patience=5,
                verbose=False)

            # Evaluate.
            for num in num_features:
                acc = gdfs.evaluate(test_dataset, num, acc_metric, 1024)
                results_dict['acc'][num] = acc
                print(f'Num = {num}, Acc = {100*acc:.2f}')
                
            # Save model
            gdfs.cpu()
            torch.save(gdfs, f'results/mnist_{args.method}_{trial}.pt')

        # Save results.
        with open(f'results/mnist_{args.method}_{trial}.pkl', 'wb') as f:
            pickle.dump(results_dict, f)
