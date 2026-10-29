import tarfile
import subprocess
import wandb
import torch
import os
import tempfile
import yaml
from datetime import datetime
import numpy as np


def get_git_hash() -> str:
    """Return the short SHA of the current HEAD commit, or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def log_individual_channels(img_tensor, prefix="train", step=0):
    """
    Log each channel of a (C, H, W) tensor to wandb.
    """
    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.detach().cpu()

    for i in range(img_tensor.shape[0]):
        channel_img = img_tensor[i]  # shape: (H, W)

        # Normalize to [0, 1] for wandb display
        min_val, max_val = channel_img.min(), channel_img.max()
        norm_img = (channel_img - min_val) / (max_val - min_val + 1e-5)

        # Convert to numpy (H, W), wandb will display it as grayscale
        wandb.log({
            f"{prefix}/channel_{i+1}": wandb.Image(norm_img.numpy(), caption=f"Channel {i+1}"),
        }, step=step)

def get_date_time():
    dt = datetime.now()
    return dt.strftime("%Y-%m-%d_%H-%M-%S")

class AssertIf:
    @staticmethod
    def _dir_exists(path: str):
        assert os.path.isdir(path), f"Directory not found: {path}"

class CheckpointManager:
    def __init__(self, run_name: str, wandb_id: str, output_dir: str = "checkpoints", subfolder: str = None):
        self.run_name = run_name
        self.wandb_id = wandb_id
        self.output_dir = output_dir
        self.subfolder = subfolder
        self.str_date_time = get_date_time()
        self.git_hash = get_git_hash()

        base_path = os.path.join(self.output_dir, f"{self.run_name}_{self.wandb_id}")
        if self.subfolder:
            self.base_dir = os.path.join(base_path, self.subfolder)
        else:
            self.base_dir = base_path

    
    def _make_run_name_dir(self):
        os.makedirs(self.base_dir, exist_ok=True)
        assert os.path.isdir(self.base_dir), f"Run directory wasn't created: {self.base_dir}"

    def _atomic_save(self, obj, final_path: str):
        """Write obj to final_path atomically and durably."""
        os.makedirs(os.path.dirname(final_path), exist_ok=True)

        # Create temp file in the same dir to ensure atomic rename
        dir_ = os.path.dirname(final_path)
        with tempfile.NamedTemporaryFile(dir=dir_, delete=False) as tmp:
            tmp_path = tmp.name
            # Write checkpoint to the temp file
            torch.save(obj, tmp)  # file-like handle -> lets us fsync
            tmp.flush()
            os.fsync(tmp.fileno())

        # Atomically replace (works on POSIX & Windows)
        os.replace(tmp_path, final_path)

        # Best-practice: fsync the directory (POSIX only; safe to try)
        try:
            dir_fd = os.open(dir_, os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            # On some systems (Windows) O_DIRECTORY isn't available; ignore.
            pass
        
    def save_config(self, config):
        path = os.path.join(self.base_dir, f"{self.str_date_time}_config.yaml")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump({
                "run_name":   self.run_name,
                "git_commit": self.git_hash,
                "config":     dict(config),
            }, f)
        art = wandb.Artifact(f"{self.run_name}_{self.wandb_id}_cfg", type="config")
        art.add_file(path)
        wandb.log_artifact(art)

    def save_checkpoint(self, model, epoch, model_class, optimizer=None, scheduler=None, loss=None, improve=False):
        if improve:
            checkpoint_path = os.path.join(f"{self.base_dir}", f"{model_class}_best_model.pth")
        else:
            checkpoint_path = os.path.join(f"{self.base_dir}", f"{self.str_date_time}_{model_class}_epoch_{epoch}.pth")
        payload = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
        }
    
        self._atomic_save(payload, checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

    def archive_checkpoints(self):
        """
        Tar.gz all .pth files from a run into a single archive.
        """
        # Use the parent directory if we're in a subfolder, otherwise use base_dir
        if self.subfolder:
            checkpoint_folder = os.path.dirname(self.base_dir)  # Go up one level from subfolder
            archive_name = f"{self.run_name}_{self.wandb_id}_{self.subfolder}_checkpoints.tar.gz"
        else:
            checkpoint_folder = self.base_dir
            archive_name = f"{self.run_name}_{self.wandb_id}_checkpoints.tar.gz"
            
        archive_path = os.path.join(self.output_dir, archive_name)
        
        assert os.path.isdir(self.base_dir), f"Missing checkpoint folder: {self.base_dir}"
        assert any(f.endswith(".pth") for f in os.listdir(self.base_dir)), "No .pth files to archive"

        with tarfile.open(archive_path, "w:gz") as tar:
            for filename in os.listdir(self.base_dir):
                if filename.endswith(".pth"):
                    file_path = os.path.join(self.base_dir, filename)
                    tar.add(file_path, arcname=filename)  # Only store filename inside archive
        
        print(f"All .pth files archived to {archive_path}")

    def save_records(self, records: np.array, phase=None):
        output_path = os.path.join(f"{self.base_dir}", f"{phase}_evaluation_results.npy")
        np.save(output_path, records)
        print(f"Results saved to {output_path}")

    def save_checkpoint_test(self, model, model_class, loss=None):
        checkpoint_path = os.path.join(f"{self.base_dir}", f"{self.str_date_time}_{model_class}_Test.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

        









def log_sample_images(images, transformed_images, label, prefix="Sample"):
    """
    Logs 5-channel grayscale images before and after transformation to wandb.
    Args:
        images: np.ndarray or torch.Tensor (H x W x C or C x H x W)
        transformed_images: torch.Tensor (C x H x W)
        label: ground truth label
    """
    if isinstance(images, torch.Tensor):
        images = images.numpy()
    if isinstance(transformed_images, torch.Tensor):
        transformed_images = transformed_images.cpu().numpy()

    if images.shape[0] == 5:
        images = np.moveaxis(images, 0, -1)
    if transformed_images.shape[0] == 5:
        transformed_images = np.moveaxis(transformed_images, 0, -1)

    for i in range(5):
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(images[:, :, i], cmap='gray')
        axes[0].set_title(f"Original - Ch{i}")
        axes[1].imshow(transformed_images[:, :, i], cmap='gray')
        axes[1].set_title(f"Transformed - Ch{i}")
        for ax in axes:
            ax.axis('off')
        wandb.log({f"{prefix}_Channel_{i}_Label_{label}": wandb.Image(plt)}, commit=False)
        plt.close(fig) 

    


