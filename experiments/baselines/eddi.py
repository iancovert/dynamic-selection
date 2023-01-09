import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from baselines.iterative import calculate_criterion, Imputer
from dynamic_selection.utils import MaskLayerGrouped


class EDDI(nn.Module):
    '''
    Efficient dynamic discovery of high-value information (EDDI): dynamic
    feature selection with missing features sampled from a conditional
    generative model.
    
    Args:
      sampler:
      predictor:
      task:
    '''
    def __init__(self, sampler, predictor, mask_layer, task='classification'):
        super().__init__()
        assert hasattr(sampler, 'impute')
        self.sampler = sampler
        self.model = predictor
        self.mask_layer = mask_layer
        assert task in ('regression', 'classification')
        self.task = task
        
        # TODO support for groups is not elegant
        if isinstance(mask_layer, MaskLayerGrouped):
            self.data_imputer = Imputer(self.mask_layer.group_matrix.cpu().data.numpy())
        else:
            self.data_imputer = Imputer()
    
    def fit(self):
        raise NotImplementedError('models should be fit beforehand')
    
    def forward(self, x, max_features, verbose=False):
        '''
        Select features and make prediction.
        
        Args:
          x:
          max_features:
          num_samples:
          verbose:
        '''
        x_masked, _ = self.select_features(x, max_features, verbose)
        return self.model(x_masked)
    
    def forward_multiple(self, x, num_features_list, verbose=False):
        '''
        Select features and make prediction for multiple feature budgets.
        
        Args:
          x:
          max_features:
          num_samples:
          verbose:
        '''
        for num, x_masked, _ in self.select_features_multiple(x, num_features_list, verbose):
            yield num, self.model(x_masked)

    def select_features(self, x, max_features, verbose=False):
        '''
        Select features.

        Args:
          x:
          max_features:
          num_samples:
          verbose:
        '''
        # Set up model.
        model = self.model
        mask_layer = self.mask_layer
        sampler = self.sampler
        data_imputer = self.data_imputer
        device = next(model.parameters()).device
        
        # Set up mask.
        if hasattr(mask_layer, 'mask_size') and (mask_layer.mask_size is not None):
            mask_size = mask_layer.mask_size
        else:
            # Must be tabular (1d data).
            assert len(x.shape) == 2
            mask_size = x.shape[1]
        num_features = mask_size
        assert 0 < max_features < num_features
        m = torch.zeros((x.shape[0], mask_size), device=device)

        for i in tqdm(range(len(x))):
            # Get row.
            x_row = x[i:i+1]
            m_row = m[i:i+1]

            for k in range(max_features):
                # Setup.
                best_ind = None
                best_criterion = - np.inf
                
                # Sample values for all remaining features.
                x_sampled = sampler.impute(x_row, m_row)[0]
                num_samples = x_sampled.shape[0]
                m_expand = m_row.repeat(num_samples, 1)
                for j in range(num_features):
                    if m[i][j] == 1:
                        # TODO support for groups is not elegant
                        if isinstance(mask_layer, MaskLayerGrouped):
                            inds = torch.where(mask_layer.group_matrix[j])[0].cpu().data.numpy()
                            original = x_row[:, inds]
                        else:
                            original = x_row[:, j]
                        x_sampled = data_imputer.impute(x_sampled, original, j)

                for j in range(num_features):
                    # Check if already included.
                    if m[i][j] == 1:
                        continue
                    
                    # Adjust mask.
                    m_expand[:, j] = 1
                    x_expand_masked = mask_layer(x_sampled, m_expand)
                
                    # Make predictions.
                    with torch.no_grad():
                        preds = model(x_expand_masked)
                    
                    # Measure criterion.
                    criterion = calculate_criterion(preds, self.task)
                    if verbose:
                        print(f'Feature {j} criterion = {criterion:.4f}')
                    
                    # Check if best.
                    if criterion > best_criterion:
                        best_criterion = criterion
                        best_ind = j
                        
                    # Turn off entry.
                    m_expand[:, j] = 0
                    
                # Select new feature.
                if verbose:
                    print(f'Selecting feature {best_ind}')
                m[i][best_ind] = 1

        # Apply mask.
        x_masked = mask_layer(x, m)
        return x_masked, m
    
    def select_features_multiple(self, x, num_features_list, verbose=False):
        '''
        Select features for multiple budgets.

        Args:
          x:
          num_features_list:
          num_samples:
          verbose:
        '''

        # Set up model.
        model = self.model
        mask_layer = self.mask_layer
        sampler = self.sampler
        data_imputer = self.data_imputer
        device = next(model.parameters()).device
        
        # Set up mask.
        if hasattr(mask_layer, 'mask_size') and (mask_layer.mask_size is not None):
            mask_size = mask_layer.mask_size
        else:
            # Must be tabular (1d data).
            assert len(x.shape) == 2
            mask_size = x.shape[1]
        num_features = mask_size
        assert isinstance(num_features_list, (list, tuple, np.ndarray))
        assert 0 < max(num_features_list) < num_features
        assert min(num_features_list) > 0
        max_features = max(num_features_list)
        m = torch.zeros((x.shape[0], mask_size), device=device)

        for k in range(max_features):
            for i in range(len(x)):
                # Get row.
                x_row = x[i:i+1]
                m_row = m[i:i+1]
                
                # Setup.
                best_ind = None
                best_criterion = - np.inf
                
                # Sample values for all remaining features.
                x_sampled = sampler.impute(x_row, m_row)[0]
                num_samples = x_sampled.shape[0]
                m_expand = m_row.repeat(num_samples, 1)
                for j in range(num_features):
                    if m[i][j] == 1:
                        # TODO support for groups is not elegant
                        if isinstance(mask_layer, MaskLayerGrouped):
                            inds = torch.where(mask_layer.group_matrix[j])[0].cpu().data.numpy()
                            original = x_row[:, inds]
                        else:
                            original = x_row[:, j]
                        x_sampled = data_imputer.impute(x_sampled, original, j)

                for j in range(num_features):
                    # Check if already included.
                    if m[i][j] == 1:
                        continue
                    
                    # Adjust mask.
                    m_expand[:, j] = 1
                    x_expand_masked = mask_layer(x_sampled, m_expand)
                
                    # Make predictions.
                    with torch.no_grad():
                        preds = model(x_expand_masked)
                    
                    # Measure criterion.
                    criterion = calculate_criterion(preds, self.task)
                    if verbose:
                        print(f'Feature {j} criterion = {criterion:.4f}')
                    
                    # Check if best.
                    if criterion > best_criterion:
                        best_criterion = criterion
                        best_ind = j
                        
                    # Turn off entry.
                    m_expand[:, j] = 0
                    
                # Select new feature.
                if verbose:
                    print(f'Selecting feature {best_ind}')
                m[i][best_ind] = 1

            # Yield current results if necessary.
            if (k + 1) in num_features_list:
                yield k + 1, mask_layer(x, m), m
    
    def evaluate(self,
                 loader,
                 max_features,
                 metric):
        '''
        Evaluate mean performance across a dataset.
        
        Args:
          loader:
          max_features:
          metric:
        '''
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # For calculating mean loss.
        pred_list = []
        label_list = []

        with torch.no_grad():
            for x, y in loader:
                # Move to GPU.
                x = x.to(device)

                # Calculate loss.
                pred = self.forward(x, max_features)
                pred_list.append(pred.cpu())
                label_list.append(y.cpu())
        
            # Calculate metric(s).
            y = torch.cat(label_list, 0)
            pred = torch.cat(pred_list, 0)
            if isinstance(metric, (tuple, list)):
                score = [m(pred, y).item() for m in metric]
            elif isinstance(metric, dict):
                score = {name: m(pred, y).item() for name, m in metric.items()}
            else:
                score = metric(pred, y).item()
                
        return score
    
    def evaluate_multiple(self,
                          loader,
                          num_features_list,
                          metric):
        '''
        Evaluate mean performance across a dataset for multiple feature budgets.
        
        Args:
          loader:
          num_features_list:
          metric:
        '''
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # For calculating mean loss.
        pred_dict = {num: [] for num in num_features_list}
        score_dict = {num: None for num in num_features_list}
        label_list = []

        with torch.no_grad():
            for x, y in loader:
                # Move to GPU.
                x = x.to(device)

                # Calculate loss.
                for num, pred in self.forward_multiple(x, num_features_list):
                    pred_dict[num].append(pred.cpu())
                label_list.append(y.cpu())
        
            # Calculate metric(s).
            y = torch.cat(label_list, 0)
            for num in num_features_list:
                pred = torch.cat(pred_dict[num], 0)
                if isinstance(metric, (tuple, list)):
                    score = [m(pred, y).item() for m in metric]
                elif isinstance(metric, dict):
                    score = {name: m(pred, y).item() for name, m in metric.items()}
                else:
                    score = metric(pred, y).item()
                score_dict[num] = score
                
        return score_dict
