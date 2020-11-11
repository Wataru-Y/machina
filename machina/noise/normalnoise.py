import numpy as np
import torch

from machina.utils import get_device
from machina.noise.base import BaseActionNoise


class NormalNoise(BaseActionNoise):

    def __init__(self, action_space, sigma=0.2):
        BaseActionNoise.__init__(self, action_space)
        self.sigma = sigma
        self.reset()

    def __call__(self, device='cpu'):
        x = np.random.normal(0, self.sigma,size=self.action_space.shape[0])

        return torch.tensor(x, dtype=torch.float, device=device)

    def reset(self):
        pass