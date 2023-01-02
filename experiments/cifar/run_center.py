import os
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import CIFAR10
from dynamic_selection import BaseModel
from dynamic_selection.utils import StaticMaskLayer2d
from resnet import ResNet18Backbone, ResNet18ClassifierHead

# Set up command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--num_trials', type=int, default=1)


# Various configurations.
crop_list = [(1,1), (1,2), (2,2), (2,3), (3,3), (3,4), (4,4),
             (4,5), (5,5), (5,6), (6,6), (6,7), (7,7)]


# Helper function for generating center crop masks.
def generate_center_mask(mask_width, width, height):
    assert width <= mask_width
    assert height <= mask_width
    mask = np.zeros((mask_width, mask_width))
    start_indx = mask_width//2-(width//2)
    start_indy = mask_width//2-(height//2)
    mask[start_indx:start_indx+width,start_indy:start_indy+height] = 1
    return mask


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
            'features': {}
        }
        acc_metric = Accuracy()

        # for num in num_features:
        for height, width in crop_list:
            # Generate random mask.
            num = height * width
            mask = generate_center_mask(8, width, height)
            mask = torch.from_numpy(mask).float()

            # Set up model.
            backbone = ResNet18Backbone()
            classifier_head = ResNet18ClassifierHead()
            mask_layer = StaticMaskLayer2d(mask, patch_size=4)
            model = nn.Sequential(mask_layer, backbone, classifier_head)

            # Train model.
            basemodel = BaseModel(model).to(device)
            basemodel.fit(
                train_dataset,
                val_dataset,
                mbsize=128,
                lr=1e-3,
                nepochs=100,
                loss_fn=nn.CrossEntropyLoss(),
                verbose=False)

            # Evaluate.
            acc = basemodel.evaluate(test_dataset, Accuracy(), 1024)
            results_dict['acc'][num] = acc
            results_dict['features'][num] = mask.cpu()
            print(f'Num = {num}, Acc = {100*acc:.2f}')

        # Save results.
        with open(f'results/cifar_center_{trial}.pkl', 'wb') as f:
            pickle.dump(results_dict, f)
