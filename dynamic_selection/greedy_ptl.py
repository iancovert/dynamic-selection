import torch
import torch.optim as optim
import pytorch_lightning as pl
from dynamic_selection.greedy import GreedyDynamicSelection
from dynamic_selection.utils import make_onehot, ConcreteSelector


class GreedyDynamicSelectionPTL(pl.LightningModule):
    '''
    Greedy adaptive feature selection.
    
    Args:
      selector:
      predictor:
      mask_layer:
      max_features:
      lr:
      loss_fn:
      val_loss_fn:
      val_loss_mode:
      factor:
      patience:
      min_lr:
      temp:
      argmax:
    '''

    def __init__(self,
                 selector,
                 predictor,
                 mask_layer,
                 max_features=None,
                 lr=1e-3,
                 loss_fn=None,
                 val_loss_fn=None,
                 val_loss_mode=None,
                 factor=0.2,
                 patience=2,
                 min_lr=1e-5,
                 temp=None,
                 argmax=False):
        # Turn off automatic optimization.
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        # Set up models and mask layer.
        self.selector = selector
        self.predictor = predictor
        self.mask_layer = mask_layer
        
        # Set up selector layer.
        self.selector_layer = ConcreteSelector()
        
        # Save optimization hyperparameters.
        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.temp = temp
        
        # Save max number of features.
        self.max_features = max_features
        
        # Save parameters for selection.
        self.argmax = argmax
        
        # Set up mask size.
        self.mask_size = mask_layer.mask_size
        
        # Save loss parameters.
        self.loss_fn = loss_fn
        if val_loss_fn is None:
            val_loss_fn = loss_fn
            val_loss_mode = 'min'
        else:
            if val_loss_mode is None:
                raise ValueError('must specify val_loss_mode (min or max) when validation_loss_fn is specified')
        self.val_loss_fn = val_loss_fn
        self.val_loss_mode = val_loss_mode
        
    def set_max_features(self, num):
        self.max_features = num
    
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
        
        # Determine mask size.
        if hasattr(self.mask_layer, 'mask_size'):
            mask_size = self.mask_layer.mask_size
        else:
            # Must be tabular (1d data).
            assert len(x.shape) == 2
            mask_size = x.shape[1]
        m = torch.zeros(len(x), mask_size, device=x.device)

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
    
    def training_step(self, batch, batch_idx):
        # Prepare optimizer.
        opt = self.optimizers()
        opt.zero_grad()

        # Set up inputs and mask.
        x, y = batch
        m = torch.zeros(len(x), self.mask_size).type_as(x)
        
        for _ in range(self.max_features):
            # Evaluate selector model.
            x_masked = self.mask_layer(x, m)
            logits = self.selector(x_masked).flatten(1)
            
            # Get selections.
            soft = self.selector_layer(logits, self.temp)
            m_soft = torch.max(m, soft)
            
            # Evaluate predictor model.
            x_masked = self.mask_layer(x, m_soft)
            pred = self.predictor(x_masked)
            
            # Calculate loss.
            loss = self.loss_fn(pred, y)
            self.manual_backward(loss / self.max_features)
            
            # Update mask, ensure no repeats.
            m = torch.max(m, make_onehot(self.selector_layer(logits - 1e6 * m, 1e-6)))
            
        # Take step.
        opt.step()
    
    def validation_step(self, batch, batch_idx):
        # Set up inputs and mask.
        x, y = batch
        m = torch.zeros(len(x), self.mask_size).type_as(x)
        
        # For mean loss.
        pred_list = []
        hard_pred_list = []
        label_list = []
        
        for _ in range(self.max_features):
            # Evaluate selector model.
            x_masked = self.mask_layer(x, m)
            logits = self.selector(x_masked).flatten(1)
            
            # Get selections, ensure no repeats.
            logits = logits - 1e6 * m
            if self.argmax:
                soft = self.selector_layer(logits, self.temp, deterministic=True)
            else:
                soft = self.selector_layer(logits, self.temp)
            m_soft = torch.max(m, soft)

            # For calculating temp = 0 loss.
            m = torch.max(m, make_onehot(soft))

            # Evaluate predictor with soft sample.
            x_masked = self.mask_layer(x, m_soft)
            pred = self.predictor(x_masked)

            # Evaluate predictor with hard sample.
            x_masked = self.mask_layer(x, m)
            hard_pred = self.predictor(x_masked)
            
            # Append results.
            pred_list.append(pred)
            hard_pred_list.append(hard_pred)
            label_list.append(y)

        return torch.cat(pred_list), torch.cat(hard_pred_list), torch.cat(label_list)
        
    
    def validation_epoch_end(self, validation_step_outputs):
        # Calculate validation loss.
        pred_list, hard_pred_list, label_list = zip(*validation_step_outputs)
        pred = torch.cat(pred_list)
        hard_pred = torch.cat(hard_pred_list)
        y = torch.cat(label_list)
        val_loss = self.val_loss_fn(pred, y)
        val_hard_loss = self.val_loss_fn(hard_pred, y)
        
        # Log results.
        self.log('val_loss', val_loss, prog_bar=True, rank_zero_only=True)
        self.log('val_hard_loss', val_hard_loss, prog_bar=True, rank_zero_only=True)
        
        # Take lr scheduler step.
        sch = self.lr_schedulers()
        sch.step(val_loss)
        
    def test_step(self, batch, batch_idx):
        # Set up inputs and mask.
        x, y = batch
        m = torch.zeros(len(x), self.mask_size).type_as(x)
        
        # For mean loss.
        pred_list = []
        label_list = []
        
        for _ in range(self.max_features):
            # Evaluate selector model.
            x_masked = self.mask_layer(x, m)
            logits = self.selector(x_masked).flatten(1)
            
            # Get selections, ensure no repeats.
            logits = logits - 1e6 * m
            m = torch.max(m, make_onehot(logits))

            # Evaluate predictor.
            x_masked = self.mask_layer(x, m)
            pred = self.predictor(x_masked)
            
            # Append results.
            pred_list.append(pred)
            label_list.append(y)
        
        # Return preds.
        return torch.cat(pred_list), torch.cat(label_list)
    
    def test_epoch_end(self, test_step_outputs):
        # Calculate validation loss.
        pred_list, label_list = zip(*test_step_outputs)
        pred = torch.cat(pred_list)
        y = torch.cat(label_list)
        test_loss = self.val_loss_fn(pred, y)
        
        # Log results.
        self.log('test_loss', test_loss, rank_zero_only=True)
    
    def configure_optimizers(self):
        opt = optim.Adam(set(list(self.predictor.parameters()) + list(self.selector.parameters())), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode=self.val_loss_mode, factor=self.factor,
                patience=self.patience, min_lr=self.min_lr, verbose=True)
        return {
            'optimizer': opt,
            'lr_scheduler': scheduler
        }
        
    def on_fit_start(self):
        # Verify parameters required for training.
        if self.loss_fn is None:
            raise ValueError('must set loss_fn for training')
        
        if self.val_loss_fn is None:
            raise ValueError('must set val_loss_fn for training')
        
        if self.val_loss_mode is None:
            raise ValueError('must set val_loss_mode for training')
        
        if self.max_features is None:
            raise ValueError('must set max_features for training')
        
    def convert(self):
        return GreedyDynamicSelection(self.selector, self.predictor, self.mask_layer)
