
"""
Feature‑extractor classes that plug into any Stable‑Baselines3 *Policy*
via the `features_extractor_class` field in `policy_kwargs`.

• R3MExtractor – frozen ResNet‑50 backbone from R3M (geometry‑aware, 2048‑D)

You can add more extractors later (VIP, VC‑1, etc.) by following the same
template.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

#                               R3M extractor                                 #
def load_r3m50() -> nn.Module:
    """Utility wrapper so we import lazily and stay Torch‑Hub free."""
    from r3m import load_r3m
    return load_r3m("resnet50")            


class R3MExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, proj_dim: int | None = None):
        super().__init__(observation_space, features_dim=2048)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.r3m = load_r3m50().eval()
        for p in self.r3m.parameters():
            p.requires_grad_(False)

        if proj_dim is not None:
            self.proj = nn.Linear(2048, proj_dim)
            self._features_dim = proj_dim
        else:
            self.proj = None
        with torch.no_grad():
            fake = torch.zeros((1,3,224,224), dtype=torch.float32, device=device)
            _ = self.r3m(fake)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:                       
            x = x.unsqueeze(0)
        x = x * 255.0                         

        with torch.no_grad():
            feat = self.r3m(x)               

        if self.proj is not None:
            feat = self.proj(feat)


        return feat


    def unfreeze_last_block(self, lr: float = 3e-5) -> Dict[str, Any]:
        """
        Unfreeze res4 parameters (last conv block) and return a param‑group
        dict you can append to the optimizer.
        """
        for name, p in self.r3m.named_parameters():
            if "layer4" in name or "res4" in name:      # name pattern in R3M
                p.requires_grad_(True)
        return {"params": [p for p in self.r3m.parameters() if p.requires_grad],
                "lr": lr}
