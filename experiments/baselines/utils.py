import torch
import torch.nn as nn
from torch.distributions import RelaxedOneHotCategorical


class ConcreteMask(nn.Module):
    '''
    For differentiable global feature selection.
    
    Args:
      num_features:
      num_select:
      group_matrix:
      append:
      gamma:
    '''

    def __init__(self, num_features, num_select, group_matrix=None, append=False, gamma=0.2):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(num_select, num_features, dtype=torch.float32))
        self.append = append
        self.gamma = gamma
        if group_matrix is None:
            self.group_matrix = None
        else:
            self.register_buffer('group_matrix', group_matrix.float())

    def forward(self, x, temp):
        dist = RelaxedOneHotCategorical(temp, logits=self.logits / self.gamma)
        sample = dist.rsample([len(x)])
        m = sample.max(dim=1).values
        if self.group_matrix is not None:
            out = x * (m @ self.group_matrix)
        else:
            out = x * m
        if self.append:
            out = torch.cat([out, m], dim=1)
        return out
    
    
class ConcreteMask2d(nn.Module):
    '''
    For differentiable global feature selection with 2d image data.
    
    Args:
      width:
      patch_size:
      num_select:
      append:
      gamma:
    '''

    def __init__(self, width, patch_size, num_select, gamma=0.2):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(num_select, width ** 2, dtype=torch.float32))
        self.upsample = torch.nn.Upsample(scale_factor=patch_size)
        self.width = width
        self.patch_size = patch_size
        self.gamma = gamma

    def forward(self, x, temp):
        dist = RelaxedOneHotCategorical(temp, logits=self.logits / self.gamma)
        sample = dist.rsample([len(x)])
        m = sample.max(dim=1).values
        m = self.upsample(m.reshape(-1, 1, self.width, self.width))
        out = x * m
        return out
