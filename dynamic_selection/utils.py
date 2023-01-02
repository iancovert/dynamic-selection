import torch
import torch.nn as nn
from torch.distributions import RelaxedOneHotCategorical


def restore_parameters(model, best_model):
    '''Move parameters from best model to current model.'''
    for param, best_param in zip(model.parameters(), best_model.parameters()):
        param.data = best_param
        
        
def generate_bernoulli_mask(shape, p):
    '''Generate binary masks with entries sampled from Bernoulli RVs.'''
    return (torch.rand(shape) < p).float()


def generate_uniform_mask(batch_size, num_features):
    '''Generate binary masks with cardinality chosen uniformly at random.'''
    unif = torch.rand(batch_size, num_features)
    ref = torch.rand(batch_size, 1)
    return (unif > ref).float()
    


def make_onehot(x):
    '''Make an approximately one-hot vector one-hot.'''
    argmax = torch.argmax(x, dim=1)
    onehot = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    onehot[torch.arange(len(x)), argmax] = 1
    return onehot


class MaskLayer(nn.Module):
    '''
    Mask layer for tabular data.
    
    Args:
      append:
      mask_size:
    '''
    def __init__(self, append, mask_size=None):
        super().__init__()
        self.append = append
        self.mask_size = mask_size

    def forward(self, x, m):
        out = x * m
        if self.append:
            out = torch.cat([out, m], dim=1)
        return out
    

MaskLayer1d = MaskLayer
    
    
class MaskLayerGrouped(nn.Module):
    '''
    Mask layer for tabular data with feature grouping.
    
    Args:
      group_matrix:
      append:
    '''
    def __init__(self, group_matrix, append):
        # Verify group matrix.
        assert torch.all(group_matrix.sum(dim=0) == 1)
        assert torch.all((group_matrix == 0) | (group_matrix == 1))
        
        # Initialize.
        super().__init__()
        self.register_buffer('group_matrix', group_matrix.float())
        self.append = append
        self.mask_size = len(group_matrix)
        
    def forward(self, x, m):
        out = x * (m @ self.group_matrix)
        if self.append:
            out = torch.cat([out, m], dim=1)
        return out
    
    
MaskLayer1dGrouped = MaskLayerGrouped


class MaskLayer2d(nn.Module):
    '''
    Mask layer for 2d image data.
    
    Args:
      append:
      mask_width:
      patch_size:
    '''

    # TODO change argument order, including in CIFAR notebooks
    def __init__(self, append, mask_width, patch_size):
        super().__init__()
        self.append = append
        self.mask_width = mask_width
        self.mask_size = mask_width ** 2
        
        # Set up upsampling.
        self.patch_size = patch_size
        if patch_size == 1:
            self.upsample = nn.Identity()
        elif patch_size > 1:
            self.upsample = nn.Upsample(scale_factor=patch_size)
        else:
            raise ValueError('patch_size should be int >= 1')

    def forward(self, x, m):
        # Reshape if necessary.
        if len(m.shape) == 2:
            m = m.reshape(-1, 1, self.mask_width, self.mask_width)
        elif len(m.shape) != 4:
            raise ValueError(f'cannot determine how to reshape mask with shape = {m.shape}')
        
        # Apply mask.
        m = self.upsample(m)
        out = x * m
        if self.append:
            out = torch.cat([out, m], dim=1)
        return out


class StaticMaskLayer1d(torch.nn.Module):
    '''
    Mask a fixed set of indices from 1d tabular data.
    
    Args:
      inds: array or tensor of indices to select.
    '''
    def __init__(self, inds):
        super().__init__()
        self.inds = inds
        
    def forward(self, x):
        return x[:, self.inds]


class StaticMaskLayer2d(torch.nn.Module):
    '''
    Mask a fixed set of pixels from 2d image data.
    
    Args:
      mask: mask indicating which parts of the image to remove at a patch level.
      patch_size: size of patches in the mask.
    '''

    def __init__(self, mask, patch_size):
        super().__init__()
        self.patch_size = patch_size
        mask = mask.float()

        # Reshape if necessary.
        if len(mask.shape) == 4:
            assert mask.shape[0] == 1
            assert mask.shape[1] == 1
        elif len(mask.shape) == 3:
            assert mask.shape[0] == 1
            mask = torch.unsqueeze(mask, 0)
        elif len(mask.shape) == 2:
            mask = torch.unsqueeze(torch.unsqueeze(mask, 0), 0)
        else:
            raise ValueError(f'unable to reshape mask with size {mask.shape}')
        assert mask.shape[-1] == mask.shape[-2]

        # Upsample mask.
        if patch_size == 1:
            mask = mask
        elif patch_size > 1:
            mask = torch.nn.Upsample(scale_factor=patch_size)(mask)
        else:
            raise ValueError('patch_size should be int >= 1')
        self.register_buffer('mask', mask)
        self.mask_size = self.mask.shape[2] * self.mask.shape[3]

    def forward(self, x):
        out = x * self.mask
        return out


class Flatten(object):
    '''Flatten image input.'''
    def __call__(self, pic):
        return torch.flatten(pic)


class ConcreteSelector(nn.Module):
    '''Output layer for selector models.'''

    def __init__(self, gamma=0.2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, temp, deterministic=False):
        if deterministic:
            # TODO this is somewhat untested, but seems like best way to preserve argmax
            return torch.softmax(logits / (self.gamma * temp), dim=-1)
        else:
            dist = RelaxedOneHotCategorical(temp, logits=logits / self.gamma)
            return dist.rsample()
