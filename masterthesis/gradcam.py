# Generate Grad-CAM utilities with enhancements:
# - channel-wise Grad-CAM
# - logging to WandB
# - separate outputs for correct and incorrect predictions


import numpy as np
import torch
import wandb
import re
from model import FindLayer
import os
from pytorch_grad_cam import GradCAM
from torchcam.utils import overlay_mask
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt


# --- requirements ---
# pip install pytorch-grad-cam
# ---------------------



class _LogitsOnly(torch.nn.Module):
    """If your model sometimes returns (logits, ch_emb), make it return logits only."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        out = self.model(x)
        return out[0] if isinstance(out, tuple) else out


class GradCAMVisualizer():

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        save_dir: str | None = None,
        log_to_wandb: bool = True,
        channel_names: list[str] | None = None,
        target_layer: torch.nn.Module | None = None,
    ):
        # wrap so GradCAM always sees logits
        self.model_raw = model
        self.model = _LogitsOnly(model).to(device).eval()

        self.device = device
        self.save_dir = save_dir
        self.log_to_wandb = log_to_wandb
        self.channel_names = channel_names

        # find target conv *on the wrapped model*
        self.target_layer = target_layer or self._find_last_conv_layer(self.model)
        if self.target_layer is None:
            raise RuntimeError("Could not find a Conv2d layer to target for Grad-CAM.")

        self.cam = GradCAM(model=self.model, target_layers=[self.target_layer])

    # ---------- helpers ----------

    @staticmethod
    def _find_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module | None:
        last = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                last = m
        return last

    @staticmethod
    def _to_rgb_background(xC: torch.Tensor, channel_index: int) -> np.ndarray:
        """
        Build a grayscale background from a single input channel and repeat to RGB.
        xC: [C,H,W] (already normalized like training)
        Returns: [H,W,3] float in [0,1]
        """
        if isinstance(xC, torch.Tensor):
            xC = xC.detach().cpu().float()  # [C,H,W]
        C, H, W = xC.shape
        assert 0 <= channel_index < C, f"channel_index out of range: {channel_index} (C={C})"

        base = xC[channel_index]  # [H,W]
        base = (base - base.min()) / (base.max() - base.min() + 1e-8)
        base_np = base.numpy().astype(np.float32)
        return np.stack([base_np, base_np, base_np], axis=-1)  # [H,W,3]

    @staticmethod
    def _to_rgb_composite(xC: torch.Tensor, mean=None, std=None, weights=None) -> np.ndarray:
        """
        Combine multi-channel input into a single grayscale base, then repeat to RGB.
        Default: simple mean over channels.
        xC: [C,H,W]
        mean/std (optional): per-channel arrays to de-normalize for nicer visuals.
        weights (optional): per-channel weights to form a weighted mean.
        Returns: [H,W,3] float in [0,1]
        """
        x = xC.detach().cpu().float().numpy()  # [C,H,W]
        C = x.shape[0]
        if mean is not None and std is not None:
            mean = np.asarray(mean).reshape(C, 1, 1)
            std  = np.asarray(std ).reshape(C, 1, 1)
            x = x * std + mean

        if weights is None:
            base = x.mean(axis=0)  # [H,W]
        else:
            w = np.asarray(weights, dtype=np.float32)
            w = w / (w.sum() + 1e-8)
            base = (x * w[:, None, None]).sum(axis=0)

        # min-max to [0,1]
        mn, mx = float(base.min()), float(base.max())
        base = (base - mn) / (mx - mn + 1e-8)
        return np.repeat(base[..., None], 3, axis=2).astype(np.float32)  # [H,W,3]

    @staticmethod
    def _sanitize_filename(s: str) -> str:
        """Keep letters, digits, dot, dash, underscore; replace others with '_'."""
        s = os.path.basename(str(s))
        return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")

    def _resolve_channel_names(self, C: int) -> list[str]:
        if self.channel_names and len(self.channel_names) == C:
            return list(self.channel_names)
        return [f"ch{c}" for c in range(C)]

    def _save_and_log(self, image_rgb_uint8: np.ndarray, filename_base: str):
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            out_path = os.path.join(self.save_dir, f"{filename_base}.png")
            plt.imsave(out_path, image_rgb_uint8)  # already RGB uint8
            print(f"Saved: {out_path}")
        if self.log_to_wandb:
            import wandb
            wandb.log({filename_base: wandb.Image(image_rgb_uint8)}, commit=False)

    @staticmethod
    def _select_target_idx(model_logits: torch.Tensor, true_label: int, pred_label: int, target: str | int) -> int:
        if isinstance(target, int):
            return int(target)
        if target == "pred":
            return int(pred_label)
        # default: "true"
        return int(true_label)

    # ---------- public APIs ----------

    def visualize(
        self,
        input_tensor: torch.Tensor,   # [C,H,W]
        true_label: int,
        pred_label: int,
        original_image_name: str,
        target_class: str | int = "true",  # "true" | "pred" | int
        mode: str = "combined",            # "combined" | "channels"
        mean=None, std=None, weights=None, # only for combined base rendering
        aug_smooth: bool = False,
        eigen_smooth: bool = False,
    ):
        """Convenience switcher."""
        if mode == "channels":
            return self.visualize_all_channels(
                input_tensor=input_tensor,
                true_label=true_label,
                pred_label=pred_label,
                original_image_name=original_image_name,
                target_class=target_class,
                aug_smooth=aug_smooth,
                eigen_smooth=eigen_smooth,
            )
        else:
            return self.visualize_combined(
                input_tensor=input_tensor,
                true_label=true_label,
                pred_label=pred_label,
                original_image_name=original_image_name,
                target_class=target_class,
                mean=mean, std=std, weights=weights,
                aug_smooth=aug_smooth,
                eigen_smooth=eigen_smooth,
            )

    def visualize_combined(
        self,
        input_tensor: torch.Tensor,   # [C,H,W]
        true_label: int,
        pred_label: int,
        original_image_name: str,
        target_class: str | int = "pred",  # "true" | "pred" | int
        mean=None, std=None, weights=None,
        aug_smooth: bool = False,
        eigen_smooth: bool = False,
    ):
        """
        One Grad-CAM heatmap over the whole multi-channel image.
        Base: grayscale composite from all channels (mean or weighted mean) repeated to RGB.
        """
        assert input_tensor.ndim == 3, f"Expected [C,H,W], got {input_tensor.shape}"
        self.model.eval()
        x = input_tensor.unsqueeze(0).to(self.device)  # [1,C,H,W]

        # pick target class
        with torch.no_grad():
            logits = self.model(x)  # [1, num_classes]
            pred = int(logits.argmax(dim=1).item())
        target_idx = self._select_target_idx(logits, true_label, pred_label if pred_label is not None else pred, target_class)
        targets = [ClassifierOutputTarget(target_idx)]

        # CAM
        with torch.enable_grad():
            grayscale_cam = self.cam(
                input_tensor=x,
                targets=targets,
                aug_smooth=aug_smooth,
                eigen_smooth=eigen_smooth,
            )[0]  # [H,W] in [0,1]

        # composite background
        bg = self._to_rgb_composite(input_tensor, mean=mean, std=std, weights=weights)  # [H,W,3] float in [0,1]
        cam_image = show_cam_on_image(bg, grayscale_cam, use_rgb=True)  # -> RGB uint8

        base_name = (
            f"GradCAM_truelabel{int(true_label)}_prediction{int(pred_label)}_"
            f"{self._sanitize_filename(original_image_name)}_combined"
        )
        self._save_and_log(cam_image, base_name)

    def visualize_all_channels(
        self,
        input_tensor: torch.Tensor,   # [C,H,W]
        true_label: int,
        pred_label: int,
        original_image_name: str,
        target_class: str | int = "true",  # "true" | "pred" | int
        aug_smooth: bool = False,
        eigen_smooth: bool = False,
    ):
        """
        Per-channel Grad-CAM for multi-channel images.

        For each channel c:
          1) zero-out other channels (isolate c),
          2) compute Grad-CAM for the chosen target class,
          3) overlay on a grayscale background made from channel c,
          4) save/log:
             'GradCAM_truelabel{y}_prediction{ŷ}_{originalimagename}_{channel}.png'
        """
        assert input_tensor.ndim == 3, f"Expected [C,H,W], got {input_tensor.shape}"
        C = int(input_tensor.shape[0])
        names = self._resolve_channel_names(C)

        base_name = (
            f"GradCAM_truelabel{int(true_label)}_prediction{int(pred_label)}_"
            f"{self._sanitize_filename(original_image_name)}"
        )

        self.model.eval()
        batched = input_tensor.unsqueeze(0).to(self.device)  # [1,C,H,W]

        # Select class
        with torch.no_grad():
            logits = self.model(batched)
            pred = int(logits.argmax(dim=1).item())
        target_idx = self._select_target_idx(logits, true_label, pred_label if pred_label is not None else pred, target_class)
        targets = [ClassifierOutputTarget(target_idx)]

        with torch.enable_grad():
            for ch in range(C):
                # 1) mask: keep only channel ch
                masked = batched.clone()
                for i in range(C):
                    if i != ch:
                        masked[0, i].zero_()

                # 2) CAM on masked input
                grayscale_cam = self.cam(
                    input_tensor=masked,
                    targets=targets,
                    aug_smooth=aug_smooth,
                    eigen_smooth=eigen_smooth,
                )[0]  # [H,W] in [0,1]

                # 3) background from this channel
                bg = self._to_rgb_background(input_tensor, channel_index=ch)  # [H,W,3]

                # 4) overlay and save/log
                cam_image = show_cam_on_image(bg, grayscale_cam, use_rgb=True)  # -> RGB uint8
                filename_base = f"{base_name}_{names[ch]}"
                self._save_and_log(cam_image, filename_base)
