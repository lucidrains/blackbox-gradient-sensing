import torch

from torch.nn import Module


class BlackboxGradientSensing(Module):
    def __init__(self):
        super().__init__()
