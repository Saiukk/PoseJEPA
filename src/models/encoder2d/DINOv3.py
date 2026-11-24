import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
from .base import BaseEncoder2D


class DINOv3(BaseEncoder2D):
    VALID_TOKENS = {"cls", "PatchToken", "ptc"}

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        layers: list = None,
        token: str = "PatchToken",
        normalize: bool = True,
        freeze: bool = True,
        image_size: int = 256,          # 256 / patch 16 = griglia 16x16, come prima
    ):
        super().__init__()

        if token not in self.VALID_TOKENS:
            raise ValueError(f"token should be one of {self.VALID_TOKENS}, received: {token}")
        if not layers:
            raise ValueError("layers non puo' essere vuoto (es. [18] per il large a 24 layer)")

        self.model_name = model_name
        self.layers = layers
        self.token = token
        self.normalize = normalize
        self.freeze = freeze
        self.image_size = image_size

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)   # submodule registrato

        cfg = self.model.config
        self.patch_size = cfg.patch_size                                  # 16
        self.n_register = int(getattr(cfg, "num_register_tokens", 0))     # 4 per i checkpoint con registri
        self.n_layers = cfg.num_hidden_layers
        if max(self.layers) > self.n_layers:
            raise ValueError(f"layer {max(self.layers)} > num_hidden_layers={self.n_layers}")
        if image_size % self.patch_size != 0:
            raise ValueError(f"image_size={image_size} non multiplo di patch_size={self.patch_size}")
        self.grid = image_size // self.patch_size

        if self.freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

        self._output_dim = None   # per il predictor

        print(f"DINOv3: {model_name}, layers={layers}, token={token}, normalize={normalize}, freeze={freeze}, image_size={image_size}")

    #@torch.no_grad()
    def extractFeatures(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Feature di DINOv3. hidden_states[0] = embeddings, hidden_states[L] = uscita del blocco L.
        Layout dei token: [CLS, registri (n_register), patch (grid*grid)].
        """
        H, W = pixel_values.shape[-2:]
        n_patch = (H // self.patch_size) * (W // self.patch_size)

        hs = self.model(pixel_values=pixel_values, output_hidden_states=True).hidden_states

        parts = []
        for L in self.layers:
            h = hs[L]                                              # (B, 1+R+N, D)
            expected = 1 + self.n_register + n_patch
            assert h.shape[1] == expected, (
                f"token inattesi: {h.shape[1]} vs {expected} "
                f"(1 CLS + {self.n_register} registri + {n_patch} patch). "
                f"Controlla layout/hidden_states della tua versione di transformers."
            )
            if self.token == "cls":
                v = h[:, 0]                                        # (B, D)
            elif self.token == "PatchToken":
                v = h[:, 1 + self.n_register:]                     # (B, N, D)  registri esclusi
            else:  # "ptc"
                v = h[:, 1 + self.n_register:].flatten(1)          # (B, N*D)
            if self.normalize:
                v = F.normalize(v, dim=-1)
            parts.append(v)

        return torch.cat(parts, dim=-1)

    @property
    def output_dim(self):
        """Output predictor"""
        if self._output_dim is None:
            device = next(self.model.parameters()).device
            dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=device)
            self._output_dim = self.extractFeatures(dummy).shape[1:]
        return self._output_dim

