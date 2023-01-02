import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from dynamic_selection.utils import restore_parameters, make_onehot, ConcreteSelector
from copy import deepcopy


class GreedyDynamicSelection(nn.Module):
    '''
    Greedy adaptive feature selection.
    
    Args:
      selector:
      predictor:
      mask_layer:
      selector_layer:
    '''

    def __init__(self, selector, predictor, mask_layer):
        super().__init__()
        
        # Set up models and mask layer.
        self.selector = selector
        self.predictor = predictor
        self.mask_layer = mask_layer
        
        # Set up selector layer.
        self.selector_layer = ConcreteSelector()

    def fit(self,
            train,
            val,
            mbsize,
            lr,
            nepochs,
            max_features,
            loss_fn,
            val_loss_fn=None,
            val_loss_mode=None,
            factor=0.2,
            patience=2,
            min_lr=1e-5,
            early_stopping_epochs=None,
            start_temp=1.0,
            end_temp=0.1,
            temp_steps=5,
            argmax=False,
            verbose=True):
        '''
        Train model to perform greedy adaptive feature selection.
        
        Args:
          train:
          val:
          mbsize:
          lr:
          nepochs:
          max_features:
          loss_fn:
          val_loss_fn:
          val_loss_mode:
          factor:
          patience:
          min_lr:
          early_stopping_epochs:
          start_temp:
          end_temp:
          temp_steps:
          argmax:
          verbose:
        '''
        # Verify arguments.
        if val_loss_fn is None:
            val_loss_fn = loss_fn
            val_loss_mode = 'min'
        else:
            if val_loss_mode is None:
                raise ValueError('must specify val_loss_mode (min or max) when validation_loss_fn is specified')
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1

        # Set up data loaders.
        train_loader = DataLoader(
            train, batch_size=mbsize, shuffle=True, pin_memory=True,
            drop_last=True, num_workers=4)
        val_loader = DataLoader(
            val, batch_size=mbsize, shuffle=False, pin_memory=True,
            drop_last=False, num_workers=4)
        
        # Set up models.
        selector = self.selector
        predictor = self.predictor
        mask_layer = self.mask_layer
        selector_layer = self.selector_layer
        device = next(predictor.parameters()).device
        val_loss_fn.to(device)
        
        # Determine mask size.
        if hasattr(mask_layer, 'mask_size') and (mask_layer.mask_size is not None):
            mask_size = mask_layer.mask_size
        else:
            # Must be tabular (1d data).
            x, y = next(iter(val))
            assert len(x.shape) == 1
            mask_size = len(x)
        
        # For tracking best models with zero temperature.
        best_val = None
        best_zerotemp_selector = None
        best_zerotemp_predictor = None
        
        # Train separately with each temperature.
        total_epochs = 0
        for temp in np.geomspace(start_temp, end_temp, temp_steps):
            if verbose:
                print(f'Starting training with temp = {temp:.4f}\n')

            # Set up optimizer and lr scheduler.
            opt = optim.Adam(set(list(predictor.parameters()) + list(selector.parameters())), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode=val_loss_mode, factor=factor, patience=patience,
                min_lr=min_lr, verbose=verbose)
            
            # For tracking best models and early stopping.
            best_selector = deepcopy(selector)
            best_predictor = deepcopy(predictor)
            num_bad_epochs = 0

            for epoch in range(nepochs):
                # Switch models to training mode.
                selector.train()
                predictor.train()

                for x, y in train_loader:
                    # Move to device.
                    x = x.to(device)
                    y = y.to(device)
                    
                    # Setup.
                    m = torch.zeros(len(x), mask_size, dtype=x.dtype, device=device)
                    selector.zero_grad()
                    predictor.zero_grad()
                    
                    for _ in range(max_features):
                        # Evaluate selector model.
                        x_masked = mask_layer(x, m)
                        logits = selector(x_masked).flatten(1)
                        
                        # Get selections.
                        soft = selector_layer(logits, temp)
                        m_soft = torch.max(m, soft)
                        
                        # Evaluate predictor model.
                        x_masked = mask_layer(x, m_soft)
                        pred = predictor(x_masked)
                        
                        # Calculate loss.
                        loss = loss_fn(pred, y)
                        (loss / max_features).backward()
                        
                        # Update mask, ensure no repeats.
                        m = torch.max(m, make_onehot(selector_layer(logits - 1e6 * m, 1e-6)))

                    # Take gradient step.
                    opt.step()
                    
                # Calculate validation loss.
                selector.eval()
                predictor.eval()
                with torch.no_grad():
                    # For mean loss.
                    pred_list = []
                    hard_pred_list = []
                    label_list = []

                    for x, y in val_loader:
                        # Move to device.
                        x = x.to(device)
                        y = y.to(device)

                        # Setup.
                        m = torch.zeros(len(x), mask_size, dtype=x.dtype, device=device)

                        for _ in range(max_features):
                            # Evaluate selector model.
                            x_masked = mask_layer(x, m)
                            logits = selector(x_masked).flatten(1)
                            
                            # Get selections, ensure no repeats.
                            logits = logits - 1e6 * m
                            if argmax:
                                soft = selector_layer(logits, temp, deterministic=True)
                            else:
                                soft = selector_layer(logits, temp)
                            m_soft = torch.max(m, soft)
                            
                            # For calculating temp = 0 loss.
                            m = torch.max(m, make_onehot(soft))
                            
                            # Evaluate predictor with soft sample.
                            x_masked = mask_layer(x, m_soft)
                            pred = predictor(x_masked)
                            
                            # Evaluate predictor with hard sample.
                            x_masked = mask_layer(x, m)
                            hard_pred = predictor(x_masked)

                            # Append predictions and labels.
                            pred_list.append(pred.cpu())
                            hard_pred_list.append(hard_pred.cpu())
                            label_list.append(y.cpu())

                    # Calculate mean loss.
                    pred = torch.cat(pred_list, 0)
                    hard_pred = torch.cat(hard_pred_list, 0)
                    y = torch.cat(label_list, 0)
                    val_loss = val_loss_fn(pred, y)
                    val_hard_loss = val_loss_fn(hard_pred, y)

                # Print progress.
                if verbose:
                    print(f'{"-"*8}Epoch {epoch+1} ({epoch + 1 + total_epochs} total){"-"*8}')
                    print(f'Val loss = {val_loss:.4f}, Zero-temp loss = {val_hard_loss:.4f}\n')

                # Update scheduler.
                scheduler.step(val_loss)

                # Check if best model.
                if val_loss == scheduler.best:
                    best_selector = deepcopy(selector)
                    best_predictor = deepcopy(predictor)
                    num_bad_epochs = 0
                else:
                    num_bad_epochs += 1

                # Check if best model with zero temperature.
                if ((best_val is None)
                    or (val_loss_mode == 'min' and val_hard_loss < best_val)
                    or (val_loss_mode == 'max' and val_hard_loss > best_val)):
                    best_val = val_hard_loss
                    best_zerotemp_selector = deepcopy(selector)
                    best_zerotemp_predictor = deepcopy(predictor)
                    
                # Early stopping.
                if num_bad_epochs > early_stopping_epochs:
                    break

            # Update total epoch count.
            if verbose:
                print(f'Stopping temp = {temp:.4f} at epoch {epoch+1}\n')
            total_epochs += (epoch + 1)

            # Copy parameters from best model.
            restore_parameters(selector, best_selector)
            restore_parameters(predictor, best_predictor)

        # Copy parameters from best model with zero temperature.
        restore_parameters(selector, best_zerotemp_selector)
        restore_parameters(predictor, best_zerotemp_predictor)

    def forward(self, x, max_features, argmax=True):
        '''
        Make predictions using selected features.

        Args:
          x:
          max_features:
          argmax:
        '''
        # Setup.
        selector = self.selector
        predictor = self.predictor
        mask_layer = self.mask_layer
        selector_layer = self.selector_layer
        device = next(predictor.parameters()).device
        
        # Determine mask size.
        if hasattr(mask_layer, 'mask_size') and (mask_layer.mask_size is not None):
            mask_size = self.mask_layer.mask_size
        else:
            # Must be tabular (1d data).
            assert len(x.shape) == 2
            mask_size = x.shape[1]
        m = torch.zeros(len(x), mask_size, device=device)

        for _ in range(max_features):
            # Evaluate selector model.
            x_masked = mask_layer(x, m)
            logits = selector(x_masked).flatten(1)

            # Update selections, ensure no repeats.
            logits = logits - 1e6 * m
            if argmax:
                m = torch.max(m, make_onehot(logits))
            else:
                m = torch.max(m, make_onehot(selector_layer(logits, 1e-6)))

        # Make predictions.
        x_masked = mask_layer(x, m)
        pred = predictor(x_masked)
        return pred, x_masked, m

    def evaluate(self,
                 dataset,
                 max_features,
                 metric,
                 batch_size,
                 argmax=True):
        '''
        Evaluate mean performance across a dataset.
        
        Args:
          dataset:
          max_features:
          metric:
          batch_size:
          argmax:
        '''
        # Setup.
        self.selector.eval()
        self.predictor.eval()
        device = next(self.predictor.parameters()).device
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
            drop_last=False, num_workers=4)

        # For calculating mean loss.
        pred_list = []
        label_list = []

        with torch.no_grad():
            for x, y in loader:
                # Move to GPU.
                x = x.to(device)

                # Calculate loss.
                pred, _, _ = self.forward(x, max_features, argmax)
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
