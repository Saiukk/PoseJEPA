import logging
import torch
import torch.nn.functional as F
from .base import BaseEncoder3D

class Uni3D(BaseEncoder3D):
    def __init__(
        self,
        model_name: str = "facebook/uni3d-base"
    ):

        super().__init__()

