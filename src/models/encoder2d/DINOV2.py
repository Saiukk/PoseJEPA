import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
from .base import BaseEncoder2D


class DINOv2(BaseEncoder2D):
    VALID_TOKENS = {"cls", "PatchToken", "ptc"}

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        layers: list = [-1],
        token: str = "ptc",
        normalize: bool = True,
        freeze: bool = True,
    ):
        super().__init__()

        if token not in self.VALID_TOKENS:
            raise ValueError(f"token should be one of {self.VALID_TOKENS}, received: {token}")

        self.model_name = model_name
        self.layers = layers
        self.token = token
        self.normalize = normalize
        self.freeze = freeze

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)  # submodule registrato

        if self.freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

        self._output_dim = None   # For the predictor

    @torch.no_grad()
    def extractFeatures(self, x) -> torch.Tensor:
        '''
        Methods to return the features computed by DINOv2
        '''
        device = next(self.model.parameters()).device
        inputs = self.processor(images=x, return_tensors="pt").to(device)
        hs = self.model(**inputs, output_hidden_states=True).hidden_states

        parts = []
        for L in self.layers:
            h = hs[L][0]
            if self.token == "cls":
                v = h[0]
                if self.normalize:
                    v = F.normalize(v, dim=-1)
            elif self.token == "PatchToken":
                v = h[1:]
                if self.normalize:
                    v = F.normalize(v, dim=-1)
            elif self.token == "ptc":
                v = h[1:].flatten()
                if self.normalize:
                    v = F.normalize(v, dim=-1)
            parts.append(v)

        return torch.cat(parts, dim=-1) if parts[0].dim() == 1 else torch.cat(parts, dim=0)

    @property
    def output_dim(self) -> int:
        '''
        Output Dimension to be used in the downstream predictor
        '''
        if self._output_dim is None:
            device = next(self.model.parameters()).device
            dummy_input = torch.zeros(3, 224, 224, device=device)  
            features = self.extractFeatures(dummy_input)
            self._output_dim = features.shape[-1]
        return self._output_dim