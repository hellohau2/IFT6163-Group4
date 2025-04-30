from __future__ import annotations

import os
import sys
import math
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt  
import cv2
import numpy as np
import torch
from PIL import Image
from image_geometry import PinholeCameraModel

# real-repo imports will be resolved dynamically after we patch sys.path


def _find_repo() -> Path:
    """Return absolute path to the *ETRG-RGS* repository."""
    here = Path(__file__).parent.resolve()
    for cand in (
        here / "ETRG-RGS",
        Path.cwd() / "ETRG-RGS",
        Path(os.getenv("ETRG_RGS_HOME", "")),
    ):
        cand = cand.expanduser().resolve()
        if cand.exists():
            return cand
    raise FileNotFoundError("ETRG-RGS repo not found.")


REPO = _find_repo()
if str(REPO) not in sys.path:
    print(f"[ETRGA] ➜ adding {REPO} to sys.path")
    sys.path.insert(0, str(REPO))

# defer heavy imports until after path patching
from utils.config import load_cfg_from_cfg_file  
from model import build_depth_mhsa               
import open_clip                                  
import clip                                       
from utils.grasp_eval import detect_grasps        

# TorchScript patch – fallback to eager RN-50                                 
_original_jit_load = torch.jit.load


def _patched_jit_load(path, *args, **kw):  
    """Gracefully load TorchScript or fallback to eager RN50 if file missing."""
    if isinstance(path, (str, os.PathLike)) and (not path or not os.path.isfile(path)):
        print(f"[ETRGA] TorchScript '{path}' not found → loading eager RN50.")
        mdl, _ = clip.load("RN50", device="cpu", jit=False)
        mdl.eval()
        return mdl
    return _original_jit_load(path, *args, **kw)


torch.jit.load = _patched_jit_load  

# Optional visual-debug helper                                                

def debug_grasp_maps(
    rgb_net: torch.Tensor,
    q: np.ndarray,
    s: np.ndarray,
    c: np.ndarray,
    w: np.ndarray,
    m: np.ndarray,
    grasps: List,
    k: int = 5,
) -> None:
    """Plot network heads and overlay *k* best grasps for qualitative checks."""
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    for ax, arr, title in zip(axs, [q, s, c, w, m], ["Q", "sin", "cos", "wid", "mask"]):
        vmin, vmax = np.nanpercentile(arr, [5, 95])
        im = ax.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title); ax.axis("off")
    fig.colorbar(im, ax=axs[-1], fraction=0.046)
    plt.tight_layout(); plt.show()

    rgb_np = (np.clip(rgb_net.permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
    for i, g in enumerate(grasps[:k]):
        v_px, u_px = (g.center if hasattr(g, "center") else g[:2])
        angle = np.degrees(g.angle if hasattr(g, "angle") else g[2])
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(rgb_np)
        ax.scatter(u_px, v_px, s=250, c="yellow", marker="+", linewidths=3, zorder=10)
        ax.set_title(f"grasp #{i}  (u={u_px:.0f}, v={v_px:.0f}, θ={angle:.1f}°)")
        ax.axis("off"); plt.tight_layout(); plt.show()


# Main wrapper                                                                


class ETRGA:
    """High-level helper that wraps the ETRG-RGS depth-MHSA model."""

    def __init__(
        self,
        repo_root: str | Path = REPO,
        cfg: str = "config/OCID-VLG/etrg_r50.yaml",
        ckpt: str = "ckpts/etrg_r50.pth",
        device: str = "cuda",
        clip_input_size: int = 224,
    ) -> None:
        """Load network weights and CLIP preprocessing."""
        repo_root = Path(repo_root).resolve()
        cfg_path, ckpt_path = repo_root / cfg, repo_root / ckpt
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"[ETRGA] loading cfg : {cfg_path}\n[ETRGA] loading ckpt: {ckpt_path}")

        self.cfg = load_cfg_from_cfg_file(str(cfg_path))
        self.device = torch.device(device)
        self.cfg.clip_pretrain = str((repo_root / "pretrain" / "RN50.pt").resolve())

        self.model, _ = build_depth_mhsa(self.cfg)
        self.model.to(self.device)

        sd = torch.load(ckpt_path, map_location="cpu")
        sd = sd.get("state_dict", sd)
        self.model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
        self.model.eval()

        self.ctx_len = int(getattr(self.cfg, "word_len", 20))
        _, self.preprocess, _ = open_clip.create_model_and_transforms("RN50", pretrained="openai", device=self.device)
        self.tokenize = open_clip.tokenize
        self.clip_input_size = clip_input_size
        self.cam_model = PinholeCameraModel()



    @torch.no_grad()
    def predict(self, img_rgb: np.ndarray, depth_img: np.ndarray, text: str) -> np.ndarray:
        """Return best-scoring grasp pose in camera coordinates."""
        h_orig, w_orig = img_rgb.shape[:2]
        target = self.clip_input_size

        # -----------------------------------------------------------------
        #  *Letter-box* the input to CLIP-square without distortion
        # -----------------------------------------------------------------
        # We resize + pad to a square *target×target* canvas while keeping the
        # original aspect ratio.  Using the *same* affine for RGB & depth means
        # their pixels stay perfectly registered.
        #
        #   scale      – uniform scaling factor that fits the longer side
        #   tx, ty     – integer translation that centre-pads the shorter side
        #   M          – 2×3 forward affine  (orig → square)
        #   M_inv      – inverse affine for later unwarping of network outputs
        # -----------------------------------------------------------------
        scale = min(target / w_orig, target / h_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        tx, ty = (target - new_w) // 2, (target - new_h) // 2
        M = np.array([[scale, 0, tx], [0, scale, ty]], np.float32)
        M_inv = cv2.invertAffineTransform(M)

        img_in   = cv2.warpAffine(img_rgb,   M, (target, target), flags=cv2.INTER_LINEAR)
        depth_in = cv2.warpAffine(depth_img, M, (target, target), flags=cv2.INTER_NEAREST)

        #  Pre-process tensors and run the network
        img_t   = self.preprocess(Image.fromarray(img_in)).unsqueeze(0).to(self.device)
        depth_t = torch.as_tensor(depth_in.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        tokens  = self.tokenize([text], context_length=self.ctx_len).to(self.device)

        dummy = torch.zeros_like(img_t[:, :1])  # unused placeholders 
        pred, _ = self.model(img_t, depth_t, tokens, *(dummy,) * 5)
        ins_mask, qua, sin, cos, wid = pred  

        #  Decode and *unwarp* prediction maps back to the original image
        qua  = torch.sigmoid(qua / 10)[0, 0].cpu().numpy()
        sin  =                sin [0, 0].cpu().numpy()
        cos  =                cos [0, 0].cpu().numpy()
        wid  =                wid [0, 0].cpu().numpy()
        mask = torch.sigmoid(ins_mask)[0, 0].cpu().numpy()

        qua_r  = cv2.warpAffine(qua,  M_inv, (w_orig, h_orig))
        sin_r  = cv2.warpAffine(sin,  M_inv, (w_orig, h_orig))
        cos_r  = cv2.warpAffine(cos,  M_inv, (w_orig, h_orig))
        wid_r  = cv2.warpAffine(wid,  M_inv, (w_orig, h_orig))
        mask_r = cv2.warpAffine(mask, M_inv, (w_orig, h_orig), flags=cv2.INTER_CUBIC)

        #  Detect top-K grasp candidates and pick the best one
        grasps, _ = detect_grasps(qua_r, sin_r, cos_r, wid_r, num_grasps=5)
        if not grasps:
            raise ValueError("No valid grasp found.")

        g0 = grasps[0]
        v_px, u_px = (g0.center if hasattr(g0, "center") else g0[:2])  # row, col
        angle      = g0.angle if hasattr(g0, "angle") else g0[2]
        Z = float(depth_img[int(v_px), int(u_px)]) 
        if Z == 0 or np.isnan(Z):                  # bad pixel
            raise ValueError("Invalid depth at grasp centre – skipping grasp")

        # De-project to 3-D (camera frame)
        ray = np.array(self.cam_model.projectPixelTo3dRay((u_px, v_px)),
                    dtype=np.float32)
        X_cam, Y_cam, Z_cam = (ray / ray[2]) * Z

        pose_cam = np.array([X_cam, Y_cam, Z_cam, 0.0, 0.0, float(angle)], dtype=np.float32)
        return pose_cam
