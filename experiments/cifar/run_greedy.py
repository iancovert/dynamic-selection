import os
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from dynamic_selection import MaskingPretrainer, GreedyDynamicSelectionPTL
from dynamic_selection.utils import MaskLayer2d
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from resnet import ResNet18Backbone, ResNet18ClassifierHead, ResNet18SelectorHead

# Set up command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--num_trials', type=int, default=1)


# Various configurations.
num_features = list(range(1, 11)) + list(range(15, 35, 5))
max_features = 20


if __name__ == '__main__':
    # Parse args.
    args = parser.parse_args()
    device = torch.device('cuda', args.gpu)
    
    # Setup for data loading.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    np.random.seed(0)
    val_inds = np.sort(np.random.choice(50000, size=10000, replace=False))
    train_inds = np.setdiff1d(np.arange(50000), val_inds)

    # Prepare datasets.
    dataset = CIFAR10('/tmp/cifar/', download=True, train=True, transform=transform_train)
    train_dataset = torch.utils.data.Subset(dataset, train_inds)
    dataset = CIFAR10('/tmp/cifar/', download=True, train=True, transform=transform_test)
    val_dataset = torch.utils.data.Subset(dataset, val_inds)
    test_dataset = CIFAR10('/tmp/cifar/', download=True, train=False, transform=transform_test)
    d_in = 32 * 32
    d_out = 10
    
    # Make results directory.
    if not os.path.exists('results'):
        os.makedirs('results')
    
    for trial in range(args.num_trials):
        # For saving results.
        results_dict = {
            'acc': {},
            'features': {},
            'model_path': None
        }
        acc_metric = Accuracy(task='multiclass', num_classes=10)
        
        # Set up networks.
        backbone = ResNet18Backbone()
        classifier_head = ResNet18ClassifierHead()
        selector_head = ResNet18SelectorHead()
        predictor = nn.Sequential(backbone, classifier_head)
        selector = nn.Sequential(backbone, selector_head)
        mask_layer = MaskLayer2d(append=False, mask_width=8, patch_size=4)

        # Pretrain predictor.
        pretrain = MaskingPretrainer(predictor, mask_layer).to(device)
        print('beginning pre-training...')
        pretrain.fit(
            train_dataset,
            val_dataset,
            mbsize=128,
            lr=1e-3,
            nepochs=100,
            loss_fn=nn.CrossEntropyLoss(),
            verbose=True)
        print('done pretraining')
        
        # Set up greedy selection object.
        gdfs = GreedyDynamicSelectionPTL(
            selector,
            predictor,
            mask_layer,
            max_features=max_features,
            loss_fn=nn.CrossEntropyLoss(),
            val_loss_fn=acc_metric,
            val_loss_mode='max',
            patience=2
        )
        
        # Set up train and val loader.
        mbsize = 128
        train_loader = DataLoader(train_dataset, batch_size=mbsize, shuffle=True,
                                  pin_memory=True, drop_last=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=mbsize, shuffle=False,
                                pin_memory=True, drop_last=False, num_workers=4)
        
        # Train with sequence of decreasing temperatures.
        best_loss = None
        best_checkpoint = None
        for temp in np.geomspace(2.0, 0.1, 5):
            # Set up callbacks and fit.
            best_soft_callback = ModelCheckpoint(
                save_top_k=1,
                monitor='val_loss',
                mode='max',
                filename='bestsoft-{epoch}-{val_loss:.4f}',
                verbose=True
            )
            best_hard_callback = ModelCheckpoint(
                save_top_k=1,
                monitor='val_hard_loss',
                mode='max',
                filename='besthard-{epoch}-{val_hard_loss:.4f}',
                verbose=True
            )
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                min_delta=0.0,
                patience=3,
                verbose=True,
                mode='max'
            )
            trainer = Trainer(
                accelerator='gpu',
                devices=[args.gpu],
                max_epochs=100,
                precision=16,
                callbacks=[best_soft_callback, best_hard_callback, early_stop_callback]
            )
            gdfs.temp = temp
            print(f'starting training with temp = {temp:.4f}')
            trainer.fit(gdfs, train_loader, val_loader)

            # Load best zero-temp epoch, check if best.
            checkpoint = best_hard_callback.best_model_path
            gdfs = GreedyDynamicSelectionPTL.load_from_checkpoint(
                checkpoint_path=checkpoint,
                selector=selector,
                predictor=predictor,
                mask_layer=mask_layer)
            gdfs.eval()
            val_loss = trainer.test(gdfs, val_loader)[0]['test_loss']
            if (best_loss is None) or (val_loss > best_loss):
                best_loss = val_loss
                best_checkpoint = checkpoint
                print(f'new best checkpoint: {best_checkpoint}')

            # Load best non-zero temp epoch for next temperature.
            checkpoint = best_soft_callback.best_model_path
            gdfs = GreedyDynamicSelectionPTL.load_from_checkpoint(
                checkpoint_path=checkpoint,
                selector=selector,
                predictor=predictor,
                mask_layer=mask_layer)

        # Load best zero-temp model.
        print('loading final model')
        results_dict['model_path'] = best_checkpoint
        gdfs = GreedyDynamicSelectionPTL.load_from_checkpoint(
            checkpoint_path=best_checkpoint,
            selector=selector,
            predictor=predictor,
            mask_layer=mask_layer)

        # Evaluate.
        gdfs_eval = gdfs.convert().to(torch.device('cuda', args.gpu)).eval()
        for num in num_features:
            acc = gdfs_eval.evaluate(test_dataset, num, acc_metric, 1024)
            results_dict['acc'][num] = acc
            print(f'Num = {num}, Acc = {100*acc:.2f}')

        # Save results.
        with open(f'results/cifar_greedy_{trial}.pkl', 'wb') as f:
            pickle.dump(results_dict, f)
