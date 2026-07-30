import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
from .base import BaseEncoder2D


class DINOv2(BaseEncoder2D):
    VALID_TOKENS = {"cls", "PatchToken", "ptc"}

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        layers: list = None,
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
    def extractFeatures(self, pixel_values: torch.Tensor) -> torch.Tensor:
        '''
        Methods to return the features computed by DINOv2
        '''
        hs = self.model(pixel_values=pixel_values, output_hidden_states=True).hidden_states

        parts = []
        for L in self.layers:
            h = hs[L]                          # (B, 1+N, D)
            if self.token == "cls":
                v = h[:, 0]                    # (B, D)
                if self.normalize:
                    v = F.normalize(v, dim=-1)
            elif self.token == "PatchToken":
                v = h[:, 1:]                   # (B, N, D)
                if self.normalize:
                    v = F.normalize(v, dim=-1)
            elif self.token == "ptc":
                v = h[:, 1:].flatten(1)        # (B, N*D)
                if self.normalize:
                    v = F.normalize(v, dim=-1)
            parts.append(v)

        return torch.cat(parts, dim=-1)

    @property
    def output_dim(self) -> int:
        '''
        Output Dimension to be used in the downstream predictor
        '''
        if self._output_dim is None:
            device = next(self.model.parameters()).device
            dummy_input = torch.zeros(1, 3, 224, 224, device=device)
            features = self.extractFeatures(dummy_input)
            self._output_dim = features.shape[1:]
        return self._output_dim