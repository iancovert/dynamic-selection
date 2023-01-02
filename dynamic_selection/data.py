import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Subset, TensorDataset


GITHUB_URL = 'https://raw.githubusercontent.com/iancovert/dynamic-selection/main/datasets/'
        
        
def data_split(dataset, val_portion=0.2, test_portion=0.2, random_state=0):
    '''
    Split dataset into train, val, test.
    
    Args:
      dataset: PyTorch dataset object.
      val_portion: percentage of samples for validation.
      test_portion: percentage of samples for testing.
      random_state: random seed.
    '''
    # Shuffle sample indices.
    rng = np.random.default_rng(random_state)
    inds = np.arange(len(dataset))
    rng.shuffle(inds)

    # Assign indices to splits.
    n_val = int(val_portion * len(dataset))
    n_test = int(test_portion * len(dataset))
    test_inds = inds[:n_test]
    val_inds = inds[n_test:(n_test + n_val)]
    train_inds = inds[(n_test + n_val):]

    # Create split datasets.
    test_dataset = Subset(dataset, test_inds)
    val_dataset = Subset(dataset, val_inds)
    train_dataset = Subset(dataset, train_inds)
    return train_dataset, val_dataset, test_dataset


def get_xy(dataset):
    '''
    Extract inputs (x) and outputs (y) from dataset object.
    
    Args:
      dataset: PyTorch dataset object.
    '''
    x, y = zip(*list(dataset))
    if isinstance(x[0], np.ndarray):
        return torch.from_numpy(np.array(x)), torch.from_numpy(np.array(y))
    elif isinstance(x[0], torch.Tensor):
        if isinstance(y[0], (int, float)):
            return torch.stack(x), torch.tensor(y)
        else:
            return torch.stack(x), torch.stack(y)
    else:
        raise ValueError(f'not sure how to concatenate data type: {type(x[0])}')


def load_spam(features=None):
    # Load data.
    data_dir = os.path.join(GITHUB_URL, 'spam.csv')
    data = pd.read_csv(data_dir)
    
    # Set features.
    if features is None:
        features = np.array([f for f in data.columns if f not in ['Outcome']])
    else:
        assert 'Outcome' not in features
        if isinstance(features, list):
            features = np.array(features)
            
    # Extract x, y.
    x = np.array(data.drop(['Outcome'], axis=1)[features]).astype('float32')
    y = np.array(data['Outcome']).astype('int64')
    
    # Create dataset object.
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    dataset.features = features
    dataset.input_size = x.shape[1]
    dataset.output_size = len(np.unique(y))
    return dataset


def load_diabetes(features=None):
    # Load data.
    data_dir = os.path.join(GITHUB_URL, 'diabetes.csv')
    data = pd.read_csv(data_dir)
    
    # Set features.
    if features is None:
        features = np.array([f for f in data.columns if f not in ['Outcome']])
    else:
        assert 'Outcome' not in features
        if isinstance(features, list):
            features = np.array(features)
            
    # Extract x, y.
    x = np.array(data.drop(['Outcome'], axis=1)[features]).astype('float32')
    y = np.array(data['Outcome']).astype('int64')
    
    # Create dataset object.
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    dataset.features = features
    dataset.input_size = x.shape[1]
    dataset.output_size = len(np.unique(y))
    return dataset


def load_miniboone(features=None):
    # Load data.
    data_dir = os.path.join(GITHUB_URL, 'miniboone.csv')
    data = pd.read_csv(data_dir)
    
    # Set features.
    if features is None:
        features = np.array([f for f in data.columns if f not in ['Outcome']])
    else:
        assert 'Outcome' not in features
        if isinstance(features, list):
            features = np.array(features)
            
    # Extract x, y.
    x = np.array(data.drop(['Outcome'], axis=1)[features]).astype('float32')
    y = np.array(data['Outcome']).astype('int64')
    
    # Create dataset object.
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    dataset.features = features
    dataset.input_size = x.shape[1]
    dataset.output_size = len(np.unique(y))
    return dataset
