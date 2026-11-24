import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseEncoder2D(nn.Module, ABC):

    def __init__(self):
        super().__init__()


    # Abstract methods to be defined in the subclasses
    @abstractmethod 
    def extractFeatures(self, x) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError

    # We can call directly encoder2d(x) to get the features
    def forward(self, x):
        return self.extractFeatures(x)

