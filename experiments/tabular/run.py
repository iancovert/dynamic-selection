import os
import sage
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import dynamic_selection as ds
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, AUROC
from captum.attr import DeepLift, IntegratedGradients
from dynamic_selection import BaseModel, MaskingPretrainer, GreedyDynamicSelection
from dynamic_selection.data import load_spam, load_diabetes, load_miniboone, data_split

import sys
sys.path.append('../')
from baselines import DifferentiableSelector, ConcreteMask
from baselines import IterativeSelector, UniformSampler
from baselines import PVAE, EDDI


# Set up command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='spam',
                    choices=['spam', 'diabetes', 'miniboone'])
parser.add_argument('--method', type=str, default='greedy',
                    choices=['sage', 'permutation', 'deeplift', 'intgrad',
                             'cae', 'iterative', 'eddi', 'greedy'])
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--num_trials', type=int, default=1)
parser.add_argument('--num_restarts', type=int, default=1)


# Various configurations.
load_data_dict = {
    'spam': load_spam,
    'diabetes': load_diabetes,
    'miniboone': load_miniboone
}
num_features_dict = {
    'spam': list(range(1, 11)) + list(range(15, 30, 5)),
    'diabetes': list(range(1, 11)),
    'miniboone': list(range(1, 11)) + list(range(15, 30, 5))
}
max_features_dict = {
    'spam': 35,
    'diabetes': 35,
    'miniboone': 35
}


# Helper function for network architecture.
def get_network(d_in, d_out):
    hidden = 128
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
    load_data = load_data_dict[args.dataset]
    num_features = num_features_dict[args.dataset]
    device = torch.device('cuda', args.gpu)
    
    # Load dataset.
    dataset = load_data()
    d_in = dataset.input_size
    d_out = dataset.output_size
    
    # Normalize and split dataset.
    mean = dataset.tensors[0].mean(dim=0)
    std = torch.clamp(dataset.tensors[0].std(dim=0), min=1e-3)
    if args.method == 'eddi':
        # PVAE generative model works better with standardized data.
        dataset.tensors = ((dataset.tensors[0] - mean) / std, dataset.tensors[1])
    else:
        dataset.tensors = (dataset.tensors[0] - mean, dataset.tensors[1])
    train_dataset, val_dataset, test_dataset = data_split(dataset)
    
    # Prepare dataloaders.
    train_loader = DataLoader(
        train_dataset, batch_size=32 if args.method == 'cae' else 128,
        shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, pin_memory=True)
    
    # Make results directory.
    if not os.path.exists('results'):
        os.makedirs('results')
    
    for trial in range(args.num_trials):
        # For saving results.
        results_dict = {
            'auroc': {},
            'acc': {},
            'features': {}
        }
        auroc_metric = lambda pred, y: AUROC(task='multiclass', num_classes=d_out)(pred.softmax(dim=1), y)
        acc_metric = Accuracy(task='multiclass', num_classes=d_out)
        
        if args.method in ['sage', 'permutation', 'deeplift', 'intgrad']:
            # Train model.
            model = get_network(d_in, d_out)
            basemodel = BaseModel(model).to(device)
            basemodel.fit(
                train_loader,
                val_loader,
                lr=1e-3,
                nepochs=250,
                loss_fn=nn.CrossEntropyLoss(),
                verbose=False)

            # Calculate feature importance scores.
            if args.method == 'sage':
                model_activation = nn.Sequential(model, nn.Softmax(dim=1))
                imputer = sage.MarginalImputer(model_activation, ds.data.get_xy(train_dataset)[0][:128].numpy())
                estimator = sage.PermutationEstimator(imputer, 'cross entropy')
                x_val, y_val = ds.data.get_xy(val_dataset)
                sage_values = estimator(x_val.numpy(), y_val.numpy(), thresh=0.01)
                ranked_features = np.argsort(sage_values.values)[::-1]

            elif args.method == 'permutation':
                permutation_importance = np.zeros(d_in)
                x_train = ds.data.get_xy(train_dataset)[0]
                for i in tqdm(range(d_in)):
                    x_val, y_val = ds.data.get_xy(val_dataset)
                    x_val[:, i] = x_train[np.random.choice(len(x_train), size=len(x_val)), i]
                    with torch.no_grad():
                        pred = model(x_val.to(device)).cpu()
                        permutation_importance[i] = - auroc_metric(pred, y_val)
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
                attr = ig.attribute(x, baselines=baseline, target=y)
                mean_abs = np.abs(attr.cpu().data.numpy()).mean(axis=0)
                ranked_features = np.argsort(mean_abs)[::-1]
            
            # Train models with top features.
            for num in num_features:
                # Prepare top features and smaller version of dataset.
                selected_features = ranked_features[:num]
                subset_dataset = load_data(dataset.features[selected_features])
                mean = subset_dataset.tensors[0].mean(dim=0)
                std = torch.clamp(subset_dataset.tensors[0].std(dim=0), min=1e-3)
                subset_dataset.tensors = (subset_dataset.tensors[0] - mean, subset_dataset.tensors[1])
                train_subset, val_subset, test_subset = data_split(subset_dataset)
                
                # Prepare subset dataloaders.
                train_subset_loader = DataLoader(train_subset, batch_size=128, shuffle=True, pin_memory=True, drop_last=True)
                val_subset_loader = DataLoader(val_subset, batch_size=1024, pin_memory=True)
                test_subset_loader = DataLoader(test_subset, batch_size=1024, pin_memory=True)
            
                best_loss = np.inf
                for _ in range(args.num_restarts):
                    # Train model.
                    model = get_network(num, d_out)
                    basemodel = BaseModel(model).to(device)
                    basemodel.fit(
                        train_subset_loader,
                        val_subset_loader,
                        lr=1e-3,
                        nepochs=250,
                        loss_fn=nn.CrossEntropyLoss(),
                        verbose=False)
                    
                    # Check if best.
                    val_loss = basemodel.evaluate(val_subset_loader, nn.CrossEntropyLoss())
                    if val_loss < best_loss:
                        best_model = basemodel
                        best_loss = val_loss
                
                # Evaluate using best model.
                auroc, acc = best_model.evaluate(test_subset_loader, (auroc_metric, acc_metric))
                results_dict['auroc'][num] = auroc
                results_dict['acc'][num] = acc
                results_dict['features'][num] = selected_features
                print(f'Num = {num}, AUROC = {100*auroc:.2f}, Acc = {100*acc:.2f}')
        
        elif args.method == 'cae':
            for num in num_features:
                # Train model with differentiable feature selection.
                model = get_network(d_in, d_out)
                selector_layer = ConcreteMask(d_in, num)
                diff_selector = DifferentiableSelector(model, selector_layer).to(device)
                diff_selector.fit(
                    train_loader,
                    val_loader,
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
            
                # Prepare smaller version of dataset.
                subset_dataset = load_data(dataset.features[selected_features])
                mean = subset_dataset.tensors[0].mean(dim=0)
                std = torch.clamp(subset_dataset.tensors[0].std(dim=0), min=1e-3)
                subset_dataset.tensors = (subset_dataset.tensors[0] - mean, subset_dataset.tensors[1])
                train_subset, val_subset, test_subset = data_split(subset_dataset)
                
                # Prepare subset dataloaders.
                train_subset_loader = DataLoader(train_subset, batch_size=128, shuffle=True, pin_memory=True, drop_last=True)
                val_subset_loader = DataLoader(val_subset, batch_size=1024, pin_memory=True)
                test_subset_loader = DataLoader(test_subset, batch_size=1024, pin_memory=True)
                
                best_loss = np.inf
                for _ in range(args.num_restarts):
                    # Train model.
                    model = get_network(num, d_out)
                    basemodel = BaseModel(model).to(device)
                    basemodel.fit(
                        train_subset_loader,
                        val_subset_loader,
                        lr=1e-3,
                        nepochs=250,
                        loss_fn=nn.CrossEntropyLoss(),
                        verbose=False)
                    
                    # Check if best.
                    val_loss = basemodel.evaluate(val_subset_loader, nn.CrossEntropyLoss())
                    if val_loss < best_loss:
                        best_model = basemodel
                        best_loss = val_loss
            
                # Evaluate using best model.
                auroc, acc = best_model.evaluate(test_subset_loader, (auroc_metric, acc_metric))
                results_dict['auroc'][num] = auroc
                results_dict['acc'][num] = acc
                results_dict['features'][num] = selected_features
                print(f'Num = {num}, AUROC = {100*auroc:.2f}, Acc = {100*acc:.2f}')
        
        elif args.method == 'iterative':
            # Train model with missingness.
            model = get_network(d_in * 2, d_out)
            mask_layer = ds.utils.MaskLayer(append=True)
            sampler = UniformSampler(ds.data.get_xy(train_dataset)[0])
            iterative = IterativeSelector(model, mask_layer, sampler).to(device)
            iterative.fit(
                train_loader,
                val_loader,
                lr=1e-3,
                nepochs=100,
                loss_fn=nn.CrossEntropyLoss(),
                patience=5,
                verbose=False)
        
            # Evaluate.
            metrics_dict = iterative.evaluate_multiple(test_loader, num_features, (auroc_metric, acc_metric))
            for num in num_features:
                auroc, acc = metrics_dict[num]
                results_dict['auroc'][num] = auroc
                results_dict['acc'][num] = acc
                print(f'Num = {num}, AUROC = {100*auroc:.2f}, Acc = {100*acc:.2f}')
        
        elif args.method == 'eddi':
            # Train PVAE.
            bottleneck = 16
            encoder = get_network(d_in * 2, bottleneck * 2)
            decoder = get_network(bottleneck, d_in)
            mask_layer = ds.MaskLayer(append=True)
            pv = PVAE(encoder, decoder, mask_layer, 128, 'gaussian').to(device)
            pv.fit(
                train_loader,
                val_loader,
                lr=1e-3,
                nepochs=250,
                verbose=False)
            
            # Train masked predictor.
            model = get_network(d_in * 2, d_out)
            sampler = UniformSampler(ds.data.get_xy(train_dataset)[0])  # TODO don't actually need sampler
            iterative = IterativeSelector(model, mask_layer, sampler).to(device)
            iterative.fit(
                train_loader,
                val_loader,
                lr=1e-3,
                nepochs=100,
                loss_fn=nn.CrossEntropyLoss(),
                patience=5,
                verbose=False)
            
            # Set up EDDI feature selection object.
            eddi_selector = EDDI(pv, model, mask_layer).to(device)
            
            # Evaluate.
            metrics_dict = eddi_selector.evaluate_multiple(test_loader, num_features, (auroc_metric, acc_metric))
            for num in num_features:
                auroc, acc = metrics_dict[num]
                results_dict['auroc'][num] = auroc
                results_dict['acc'][num] = acc
                print(f'Num = {num}, AUROC = {100*auroc:.2f}, Acc = {100*acc:.2f}')
        
        elif args.method == 'greedy':
            # Prepare networks.
            predictor = get_network(d_in * 2, d_out)
            selector = get_network(d_in * 2, d_in)
            
            # Pretrain predictor.
            mask_layer = ds.utils.MaskLayer(append=True)
            pretrain = MaskingPretrainer(predictor, mask_layer).to(device)
            pretrain.fit(
                train_loader,
                val_loader,
                lr=1e-3,
                nepochs=100,
                loss_fn=nn.CrossEntropyLoss(),
                patience=5,
                verbose=False)
            
            # Train selector and predictor jointly.
            gdfs = GreedyDynamicSelection(selector, predictor, mask_layer).to(device)
            gdfs.fit(
                train_loader,
                val_loader,
                lr=1e-3,
                nepochs=250,
                max_features=max_features_dict[args.dataset],
                loss_fn=nn.CrossEntropyLoss(),
                patience=5,
                verbose=False)
            
            # Evaluate.
            for num in num_features:
                auroc, acc = gdfs.evaluate(test_loader, num, (auroc_metric, acc_metric))
                results_dict['auroc'][num] = auroc
                results_dict['acc'][num] = acc
                print(f'Num = {num}, AUROC = {100*auroc:.2f}, Acc = {100*acc:.2f}')
                
            # Save model
            gdfs.cpu()
            torch.save(gdfs, f'results/{args.dataset}_{args.method}_{trial}.pt')
        
        # Save results.
        with open(f'results/{args.dataset}_{args.method}_{trial}.pkl', 'wb') as f:
            pickle.dump(results_dict, f)
