# -- Block TensorFlow import to prevent protobuf crash (train.py never uses TF) --
import sys, types, importlib.machinery
_tf = types.ModuleType("tensorflow"); _tf.__version__ = "0.0.0"
_tf.__spec__ = importlib.machinery.ModuleSpec("tensorflow", None)
sys.modules["tensorflow"] = _tf

import argparse
import itertools
import os
import re
import tempfile
import torch
import pandas as pd
import numpy as np
import wandb
import yaml
import random
from tqdm import tqdm
import shutil
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import tarfile
import matplotlib.pyplot as plt
from gradcam import GradCAMVisualizer
from model import LinearOnEmbeddings, ResNet50_Modified, MultiChannelResNet50, OpenPhenomMAE
from dataset import CellPaintingDataset, FiveChannelAlbumentations, BuildIndex, \
    SplitManager, MiniDataset, TVNEmbeddingDataset, SyntheticDataset, WithIndex
import schedulefree
from topK import TopKSelector
from mm_logging import CheckpointManager
from config.check import validate_config


def parse_args():
    """Parse command line arguments with defaults from wandb_config.yaml."""
    # Load defaults from YAML
    default_config_path = os.path.join(os.path.dirname(__file__), "config", "wandb_config.yaml")
    defaults = {}
    if os.path.exists(default_config_path):
        with open(default_config_path) as f:
            defaults = yaml.safe_load(f) or {}

    parser = argparse.ArgumentParser(
        description="Train cell painting classification models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config file (overrides defaults if provided)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (overrides defaults)")

    # Data settings
    parser.add_argument("--mean-std", type=str, default=defaults.get("mean_std", "CELLPAINTING_3_w_NEG"),
                        choices=["IMAGENET", "CELLPAINTING_1", "CELLPAINTING_1_w_NEG", "CELLPAINTING_3", "CELLPAINTING_3_w_NEG", "MAE", "CUSTOM"],
                        help="Normalization statistics to use")
    parser.add_argument("--source-types", type=str, default=defaults.get("source_types", "source_3_w_neg"),
                        help="Source type for data loading")
    parser.add_argument("--index", type=str, default=defaults.get("index", "all_sources.pq"),
                        help="Index file for data")
    parser.add_argument("--image-path", type=str, default=defaults.get("image_path", "masterthesis/source_3/"),
                        help="Path to image directory")

    # Mode settings
    parser.add_argument("--debug-mode", action="store_true", default=defaults.get("debug_mode", False),
                        help="Enable debug mode with reduced dataset")
    parser.add_argument("--synthetic", action="store_true", default=defaults.get("synthetic", False),
                        help="Use synthetic dataset for testing")
    parser.add_argument("--embedding-mode", action="store_true", default=defaults.get("embedding_mode", False),
                        help="Train linear probe on pre-computed embeddings")
    parser.add_argument("--inference", action="store_true", default=defaults.get("inference", False),
                        help="Run inference on test set only")
    parser.add_argument("--gradcam", action="store_true", default=defaults.get("gradcam", False),
                        help="Generate Grad-CAM visualizations")
    parser.add_argument("--records", action="store_true", default=defaults.get("records", False),
                        help="Save prediction records for statistics")
    parser.add_argument("--return-channelwise-embeddings", type=lambda x: x.lower() == 'true',
                        default=defaults.get("return_channelwise_embeddings", False),
                        help="Extract and save channel-wise embeddings (true/false)")

    # Model settings
    parser.add_argument("--architecture", type=str, default=defaults.get("architecture", "ResNet50_Modified"),
                        choices=["ResNet50_Modified", "OpenPhenomMAE", "MultiChannelResNet50"],
                        help="Model architecture to use")
    parser.add_argument("--mode", type=str, default=defaults.get("mode", "combined"),
                        choices=["combined", "channel-wise"],
                        help="Grad-CAM mode")
    parser.add_argument("--pretrained", action="store_true", default=defaults.get("pretrained", True),
                        help="Use pretrained weights")
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained",
                        help="Do not use pretrained weights")
    parser.add_argument("--freeze", action="store_true", default=defaults.get("freeze", False),
                        help="Freeze encoder weights")
    parser.add_argument("--no-freeze", action="store_false", dest="freeze",
                        help="Do not freeze encoder weights")

    # Checkpoint/resume settings
    parser.add_argument("--resume", action="store_true", default=defaults.get("resume", False),
                        help="Resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=defaults.get("checkpoint", ""),
                        help="Path to checkpoint file for resuming")
    parser.add_argument("--embeddings-path", type=str, default=defaults.get("embeddings_path", ""),
                        help="Path to embeddings .npz file (embedding mode)")

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 64),
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 100),
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=defaults.get("lr", 0.001),
                        help="Learning rate")
    parser.add_argument("--lr-scheduler", type=str, default=defaults.get("lr_scheduler", "auto"),
                        choices=["auto", "cosine", "step"],
                        help="Learning rate scheduler")
    parser.add_argument("--weight-decay", type=float, default=defaults.get("weight_decay", 0.01),
                        help="Weight decay (L2 regularization)")
    parser.add_argument("--label-smoothing", type=float, default=defaults.get("label_smoothing", 0.1),
                        help="Label smoothing for cross-entropy loss (0 = off)")
    parser.add_argument("--lr-warmup-epochs", type=int, default=defaults.get("lr_warmup_epochs", 5),
                        help="Linear warmup epochs before cosine decay (cosine scheduler only)")
    parser.add_argument("--early-stopping-metric", type=str, default=defaults.get("early_stopping_metric", "Accuracy"),
                        choices=["Accuracy", "Validation Loss"],
                        help="Metric for early stopping")
    parser.add_argument("--early-stopping-patience", type=int, default=defaults.get("early_stopping_patience", 10),
                        help="Patience for early stopping")

    # Data splits
    parser.add_argument("--splits", type=str, default=defaults.get("splits", "batches"),
                        choices=["random", "wells", "plates", "batches"],
                        help="Split strategy for train/val/test")
    parser.add_argument("--eval-split", type=float, default=defaults.get("eval_split", 0.2),
                        help="Fraction of data for validation")
    parser.add_argument("--test-split", type=float, default=defaults.get("test_split", 0.1),
                        help="Fraction of data for test")
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 12),
                        help="Random seed")

    # System settings
    parser.add_argument("--num-workers", type=int, default=defaults.get("num_workers", 8),
                        help="Number of data loader workers")

    # Augmentation settings (nested in config)
    aug_defaults = defaults.get("augmentation", {})
    parser.add_argument("--resize", type=int, nargs=2, default=aug_defaults.get("resize", [224, 224]),
                        help="Resize dimensions [H, W]")
    parser.add_argument("--horizontal-flip-prob", type=float, default=aug_defaults.get("horizontal_flip_prob", 0.5),
                        help="Horizontal flip probability")
    parser.add_argument("--vertical-flip-prob", type=float, default=aug_defaults.get("vertical_flip_prob", 0.5),
                        help="Vertical flip probability")
    parser.add_argument("--rotation-prob", type=float, default=aug_defaults.get("rotation_prob", 0),
                        help="Rotation probability")
    parser.add_argument("--noise-std", type=float, default=aug_defaults.get("noise_std", 0.2),
                        help="Noise standard deviation")
    parser.add_argument("--noise-prob", type=float, default=aug_defaults.get("noise_prob", 0.5),
                        help="Noise probability")
    parser.add_argument("--brightness-contrast-prob", type=float, default=aug_defaults.get("brightness_contrast_prob", 0.5),
                        help="Brightness/contrast adjustment probability")
    parser.add_argument("--blur-prob", type=float, default=aug_defaults.get("blur_prob", 0.0),
                        help="Gaussian blur probability (kernel 3-7px)")
    parser.add_argument("--coarse-dropout-prob", type=float, default=aug_defaults.get("coarse_dropout_prob", 0.0),
                        help="CoarseDropout (Cutout) probability — randomly blacks out 1-8 patches")

    args = parser.parse_args()
    return args


def args_to_config(args):
    """Convert argparse namespace to config dict matching wandb_config.yaml structure."""
    # Load base config: --config if given, otherwise the default wandb_config.yaml.
    # This preserves nested augmentation keys (e.g. RandomResizedCrop, plasma_shadow)
    # that are not exposed as CLI arguments and would otherwise be dropped.
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    else:
        default_path = os.path.join(os.path.dirname(__file__), "config", "wandb_config.yaml")
        config = {}
        if os.path.exists(default_path):
            with open(default_path) as f:
                config = yaml.safe_load(f) or {}

    # Preserve nested augmentation keys from YAML (e.g. RandomResizedCrop, plasma_shadow);
    # CLI flat args override only the keys they know about.
    aug_base = config.get("augmentation", {})

    # Map CLI args to config (CLI args take precedence)
    cli_config = {
        "mean_std": args.mean_std,
        "debug_mode": args.debug_mode,
        "synthetic": args.synthetic,
        "resume": args.resume,
        "embedding_mode": args.embedding_mode,
        "gradcam": args.gradcam,
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "embeddings_path": args.embeddings_path,
        "records": args.records,
        "batch_size": args.batch_size,
        "eval_split": args.eval_split,
        "test_split": args.test_split,
        "num_workers": args.num_workers,
        "return_channelwise_embeddings": args.return_channelwise_embeddings,
        "inference": args.inference,
        "lr": args.lr,
        "lr_scheduler": args.lr_scheduler,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "lr_warmup_epochs": args.lr_warmup_epochs,
        "architecture": args.architecture,
        "early_stopping_metric": args.early_stopping_metric,
        "early_stopping_patience": args.early_stopping_patience,
        "epochs": args.epochs,
        "splits": args.splits,
        "seed": args.seed,
        "index": args.index,
        "source_types": args.source_types,
        "image_path": args.image_path,
        "pretrained": args.pretrained,
        "freeze": args.freeze,
        "augmentation": {
            **aug_base,                                              # nested keys (RandomResizedCrop, plasma_shadow, …)
            "resize": args.resize,                                   # CLI overrides for flat keys below
            "horizontal_flip_prob": args.horizontal_flip_prob,
            "vertical_flip_prob": args.vertical_flip_prob,
            "rotation_prob": args.rotation_prob,
            "noise_std": args.noise_std,
            "noise_prob": args.noise_prob,
            "brightness_contrast_prob": args.brightness_contrast_prob,
            "blur_prob": args.blur_prob,
            "coarse_dropout_prob": args.coarse_dropout_prob,
        },
    }

    # Merge: config file values are overwritten by CLI values
    config.update(cli_config)
    return config


        
class BuildComponents:
    def __init__(self, config: dict):
        self.wandb_id = wandb.run.id
        self.wandb_name = wandb.run.name
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_class = self.config["architecture"]
        self._set_seed(self.config["seed"])
        self.df = None
        self.image_path = self.config["image_path"]

    def _set_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True

    def index(self):
        if self.config.get("synthetic", False):
            print("Using synthetic dataset")
            self.df = pd.DataFrame({"pert_iname": list(range(8))})

        elif self.config.get("embedding_mode", False):
            emb_path = self.config.get("embeddings_path", None)
            assert emb_path is not None and os.path.exists(emb_path), f"Missing embeddings file: {emb_path}"

            npz = np.load(emb_path, allow_pickle=True)
            X = npz["emb"]
            names = npz["names"].astype(str)

            # Load metadata and align by sample ID
            meta_path = self.config["source_types"] + ".pq"
            meta = pd.read_parquet(meta_path)
            meta["Metadata_Sample_ID"] = meta["Metadata_Sample_ID"].astype(str)
            meta_idx = meta.set_index("Metadata_Sample_ID", drop=False)
            aligned = meta_idx.reindex(names).reset_index(drop=True)

            # Build DataFrame: embedding columns + metadata
            emb_cols = [str(i) for i in range(X.shape[1])]
            self.df = pd.DataFrame(X, columns=emb_cols)
            self.df["Metadata_Sample_ID"] = names
            for col in ["Metadata_Batch", "Metadata_Plate", "Metadata_Well",
                        "pert_iname", "pert_iname_str"]:
                if col in aligned.columns:
                    self.df[col] = aligned[col].values

            # Add lowercase aliases
            alias = {"Metadata_Batch": "batch", "Metadata_Plate": "plate", "Metadata_Well": "well"}
            for hi, lo in alias.items():
                if hi in self.df.columns:
                    self.df[lo] = self.df[hi]

            if "pert_iname_str" in self.df.columns:
                self.df["is_control"] = self.df["pert_iname_str"].eq("DMSO")

            if "Metadata_Sample_ID" not in self.df.columns:
                self.df["Metadata_Sample_ID"] = np.arange(len(self.df)).astype(str)
            else:
                self.df["Metadata_Sample_ID"] = self.df["Metadata_Sample_ID"].astype(str)

            if "pert_iname" in self.df.columns:
                self.df["pert_iname"] = self.df["pert_iname"].astype(int)

            print(f"Loaded transformed embeddings from {emb_path}")

        else:
            path = self.config["source_types"] + ".pq"
            
            if os.path.exists(path):
                print(f"Loading index from {path}")
                self.df = pd.read_parquet(path, engine="fastparquet")
                if "Metadata_Sample_ID" in self.df.columns:
                    self.df["Metadata_Sample_ID"] = self.df["Metadata_Sample_ID"].astype(str)
            else:
                print(f"Index file {path} not found, building index from scratch.")
                self.df = BuildIndex(self.config["source_types"], self.config["index"]).dataset
                assert self.df is not None and len(self.df) > 0, "BuildIndex returned empty dataframe"
            # Defensive: drop rows whose image file is missing on disk. Incomplete
            # staging (a few un-synced images) otherwise kills the whole run on the
            # first absent file, hours in. No-op when every image is present.
            img_dir = self.config.get("image_path")
            if img_dir and os.path.isdir(img_dir) and "Metadata_Sample_ID" in self.df.columns:
                have = set(os.listdir(img_dir))
                keep = (self.df["Metadata_Sample_ID"].astype(str) + ".jpg").isin(have)
                n_missing = int((~keep).sum())
                if n_missing:
                    print(f"[index] {n_missing}/{len(self.df)} rows reference missing "
                          f"images on disk — dropping them")
                    self.df = self.df[keep].reset_index(drop=True)
            assert "pert_iname" in self.df.columns, "pert_iname not in DataFrame"
        return self.df
    
    def dataset(self):
        return self.df if self.df is not None else self.index()

    def dataloaders(self, train_keys, eval_keys, test_keys):
        num_workers = self.config.get("num_workers", 0)
        batch_size  = self.config.get("batch_size", 32)

        if self.config.get("embedding_mode", False) and not self.config.get("inference", False):
            embedding_cols = sorted(
                [str(c) for c in self.df.columns if re.fullmatch(r"\d+", str(c))],
                key=lambda x: int(x)  # ensure numeric order: 0,1,2,...,10 (not 0,1,10,2)
            )

            DatasetClass = TVNEmbeddingDataset
            train_ds = DatasetClass(train_keys, embedding_cols=embedding_cols) if train_keys is not None else None
            eval_ds  = DatasetClass(eval_keys,  embedding_cols=embedding_cols) if eval_keys  is not None else None
            test_ds  = DatasetClass(test_keys,  embedding_cols=embedding_cols) if test_keys  is not None else None

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                    num_workers=num_workers, persistent_workers=(num_workers > 0), pin_memory=True) if train_ds else None
            eval_loader  = DataLoader(eval_ds,  batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, persistent_workers=(num_workers > 0), pin_memory=True) if eval_ds else None
            test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, persistent_workers=(num_workers > 0), pin_memory=True) if test_ds else None
            return train_loader, eval_loader, test_loader
        
        elif self.config.get("embedding_mode", False) and self.config.get("inference", False):
            # Build ONLY the test dataset/loader for TVN inference

            embedding_cols = sorted(
                [str(c) for c in self.df.columns if re.fullmatch(r"\d+", str(c))],
                key=lambda x: int(x)  # ensure numeric order: 0,1,2,...,10 (not 0,1,10,2)
            )

            DatasetClass = TVNEmbeddingDataset
            test_dataset = DatasetClass(test_keys, embedding_cols=embedding_cols)



            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, persistent_workers=(num_workers > 0), pin_memory=True
            )
            return None, None, test_loader

        if self.config.get("synthetic", False):
            print("Using synthetic dataset")
            train_dataset = SyntheticDataset(10000, 8, [5, 224, 224])
            eval_dataset  = SyntheticDataset(1000, 8, [5, 224, 224])
            test_dataset  = SyntheticDataset(1000, 8, [5, 224, 224])


        else: 
            print("Using CellPaintingDataset")
            # --- Normalization wiring ----------------------------------------
            # FiveChannelAlbumentations reads `mean_std` from the dict it is
            # given; `mean_std` lives at the TOP level of the config, not inside
            # `augmentation`, so it must be propagated into the transform dict.
            #   * Fresh run  -> honor the configured `mean_std`.
            #   * Resuming an existing checkpoint -> KEEP the CELLPAINTING_1
            #     fallback: the existing checkpoints were trained under it, and
            #     normalization must match the trained weights. Re-extracting
            #     embeddings with a different mean_std would corrupt them
            #     (train/inference mismatch).
            _aug_cfg = dict(self.config["augmentation"])
            if not self.config.get("resume", False):
                _aug_cfg["mean_std"] = self.config.get("mean_std", "CELLPAINTING_1")
            print(f"🎨 Normalization in use: mean_std = "
                  f"{_aug_cfg.get('mean_std', 'CELLPAINTING_1 (resume fallback)')}")
            transform_train = FiveChannelAlbumentations(_aug_cfg, "train")
            transform_test = FiveChannelAlbumentations(_aug_cfg, "test")
            
            DatasetClass = MiniDataset if self.config.get("debug_mode", False) else CellPaintingDataset
            print("⚠️  Debug mode activated: Using MiniDataset with reduced sample size.") if self.config.get("debug_mode", False) else print("Using full dataset.")

            train_dataset = DatasetClass(self.image_path, train_keys, transform=transform_train)
            eval_dataset = DatasetClass(self.image_path, eval_keys, transform=transform_test)
            test_dataset = DatasetClass(self.image_path, test_keys, transform=transform_test)

        image_mode = not (self.config.get("embedding_mode", False) or self.config.get("synthetic", False))
        if image_mode:
            train_dataset = WithIndex(train_dataset)
            eval_dataset  = WithIndex(eval_dataset) 
            test_dataset = WithIndex(test_dataset)   

        print(f"Creating DataLoaders with batch size {batch_size} and {num_workers} workers")
        train_loader = DataLoader(train_dataset, 
                                  batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=(num_workers > 0), pin_memory=True)
        eval_loader = DataLoader(eval_dataset, 
                                 batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=(num_workers > 0), pin_memory=True)
        test_loader = DataLoader(test_dataset, 
                                 batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=(num_workers > 0), pin_memory=True)

        for name, dl in [("train", train_loader), ("eval", eval_loader), ("test", test_loader)]:
            assert len(dl) > 0, f"{name} DataLoader is empty"
            if getattr(dl, "persistent_workers", False):
                assert dl.num_workers > 0, "persistent_workers=True but num_workers=0"
        return train_loader, eval_loader, test_loader
    
    def model(self):
        if self.config.get("synthetic", False):
            num_classes = 7
        else: 
            num_classes = len(self.df["pert_iname"].unique())

        if self.config.get("embedding_mode", False):
            print(f"Using LinearOnEmbeddings")
            embedding_columns = [col for col in self.df.columns if re.fullmatch(r"\d+", str(col))]
            in_dim = len(embedding_columns)
            return LinearOnEmbeddings(in_dim, num_classes).to(self.device)
        
        arch = self.config.get("architecture", "ResNet50_Modified")
        if arch == "ResNet50_Modified":
            print("Using ResNet50_Modified architecture")
            model = ResNet50_Modified(num_classes, pretrained= self.config["pretrained"], freeze_encoder=self.config["freeze"], return_channelwise_embeddings=self.config["return_channelwise_embeddings"], embedding_mode="joint")
        elif arch == "MultiChannelResNet50":
            print("Using MultiChannelResNet50 architecture")
            model = MultiChannelResNet50(num_classes)
        elif arch == "OpenPhenomMAE":
            print("Using OpenPhenomMAE architecture")
            model = OpenPhenomMAE(num_classes=num_classes, freeze_encoder=self.config["freeze"], return_channelwise_embeddings=self.config["return_channelwise_embeddings"])
        else:
            raise ValueError(f"Unknown architecture {arch}")
        return model.to(self.device)

    def optimizer(self, model):
        weight_decay    = self.config.get("weight_decay", 0.0)
        warmup_epochs   = self.config.get("lr_warmup_epochs", 0)

        if self.config["lr_scheduler"] == "auto":
            optimizer = schedulefree.AdamWScheduleFree(
                model.parameters(),
                lr=self.config["lr"],
                weight_decay=weight_decay,
            )
            return optimizer, None
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config["lr"],
                weight_decay=weight_decay,
            )
            if self.config["lr_scheduler"] == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            elif self.config["lr_scheduler"] == "cosine":
                if warmup_epochs > 0:
                    warmup = torch.optim.lr_scheduler.LinearLR(
                        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
                    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=self.config["epochs"] - warmup_epochs)
                    scheduler = torch.optim.lr_scheduler.SequentialLR(
                        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
                else:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=self.config["epochs"])
            else:
                scheduler = None
            return optimizer, scheduler

    def criterion(self):
        return torch.nn.CrossEntropyLoss(
            label_smoothing=self.config.get("label_smoothing", 0.0)
        )


class Trainer:
    def __init__(self, model, optimizer, criterion, device, config, run_name, wandb_id, ckpt=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.run_name = run_name
        self.wandb_id = wandb_id
        self.scheduler = None
        self.early_stopping_metric = config.get("early_stopping_metric", "Validation Loss")
        self.early_stopping_patience = config.get("early_stopping_patience", 5)
        self.best_metric = None
        self.epochs_since_improvement = 0
        self.ckpt = ckpt


    @staticmethod
    def _assert_cuda_batch(x):
        assert x.is_cuda, f"Batch not on CUDA: {x.device}"


    def training(self, train_loader, epoch, num_epochs):
        self.model.train()
        if self.config["lr_scheduler"] == "auto":
            self.optimizer.train()
        else:
            if self.scheduler:
                self.scheduler.step()
        running_loss = 0.0
        is_image_mode = not (self.config.get("embedding_mode", False) or self.config.get("synthetic", False))

        for batch_idx, (images, labels, *_) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")):
            if is_image_mode:
                images = images.to(self.device, non_blocking=True).to(memory_format=torch.channels_last)
            else:   
                images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self._assert_cuda_batch(images)
            if is_image_mode:
                assert images.dim() == 4, f"Expected NCHW, got {tuple(images.shape)}"
                assert images.size(1) == 5, f"Expected 5 channels, got {images.size(1)}"
            assert labels.dtype in (torch.int64, torch.int32), f"Labels must be integer class ids, got {labels.dtype}"
            self.optimizer.zero_grad()
            outputs = self.model(images)
            if isinstance(outputs, tuple):
                logits, _ = outputs
            else:
                logits = outputs
            assert logits.dim() == 2, f"Expected (N, C) logits, got {tuple(logits.shape)}"
            assert logits.size(0) == labels.size(0), "Batch size mismatch between logits and labels"
            assert torch.isfinite(logits).all(), "Model produced NaN/Inf logits"
            with torch.no_grad():
                assert labels.max().item() < logits.size(1), f"Label {labels.max().item()} out of range for {logits.size(1)} classes"
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {avg_loss:.4f}")
        wandb.log({"Epoch": epoch, "Train Loss": avg_loss, "Learning Rate": self.optimizer.param_groups[0]['lr']})

        if epoch % 5 == 0 and epoch != 0:
            print(f"Saving checkpoint at epoch {epoch}")
            self.ckpt.save_checkpoint(self.model, epoch, self.config["architecture"], self.optimizer, loss=avg_loss)

    def evaluate(self, loader, epoch, num_epochs, phase="Validation", gradcam=False, save_embeddings=False):
        self.model.eval()
        if self.config["lr_scheduler"] == "auto":
            self.optimizer.eval()

        correct = total = 0
        loss_total = 0.0
        all_preds, all_labels = [], []
        record_array = []
   
        if self.config.get("return_channelwise_embeddings", False):
            print("Returning channelwise embeddings")
        emb_list, name_list, id_list, ytrue_list, ypred_list = [], [], [], [], []

        selector = TopKSelector(k_correct=5, k_error=3)

        def _to_np1d(x, dtype=None):
            if isinstance(x, torch.Tensor):
                a = x.detach().cpu().numpy()
                return a.astype(dtype) if dtype is not None else a
            if isinstance(x, pd.Series):
                return x.to_numpy(copy=False, dtype=dtype)
            try:
                return np.asarray(list(x), dtype=dtype)
            except TypeError:
                return np.asarray(x, dtype=dtype)

        with torch.no_grad():
            for batch in loader:
                if len(batch) < 7:
                    images, labels_t, is_ctrl, plate_ids, wells, batch_ids = batch
                    idxs = None
                else:
                    images, labels_t, is_ctrl, plate_ids, wells, batch_ids, idxs = batch

                images = images.to(self.device)
                labels_t = labels_t.to(self.device)

                outputs = self.model(images)                # [B, C]
                if isinstance(outputs, tuple):
                    logits, emb = outputs
                else:
                    logits = outputs
                    emb = None
                
                loss = self.criterion(logits, labels_t)
                preds_t = logits.argmax(dim=1)             # [B]

                correct += (preds_t == labels_t).sum().item()
                total   += labels_t.size(0)
                loss_total += loss.item()

                probs_t = torch.softmax(logits, dim=1)                         # [B, C]
                conf_pred_t = probs_t.gather(1, preds_t.unsqueeze(1)).squeeze(1)  # [B] prob of predicted class

                preds_np     = preds_t.detach().cpu().numpy()
                labels_np    = labels_t.detach().cpu().numpy()
                conf_class = conf_pred_t.detach().cpu().numpy()
                conf = probs_t.max(dim=1).values.detach().cpu().numpy()  
                logits_np    = logits.detach().cpu().numpy()

                is_ctrl_np   = _to_np1d(is_ctrl,   dtype=np.int8)
                plate_ids_np = _to_np1d(plate_ids, dtype=object)
                well_ids_np  = _to_np1d(wells,     dtype=object)
                batch_ids_np = _to_np1d(batch_ids)  # int64 usually

                all_preds.extend(preds_np.tolist())
                all_labels.extend(labels_np.tolist())

                errors_np = (preds_np != labels_np).astype(int)
                batch_records = np.column_stack([
                    plate_ids_np, well_ids_np, is_ctrl_np, batch_ids_np,
                    labels_np, preds_np, conf, conf_class, errors_np, logits_np
                ])

                if emb is not None:
                    base_ds = loader.dataset.base if hasattr(loader.dataset, "base") else loader.dataset
                    if idxs is not None:
                        idxs_np = idxs.detach().cpu().numpy()
                    else:
                        # Fallback if your loader doesn't return indices:
                        # just create 0..B-1 for this batch. (OK for saving in-order,
                        # but not useful for re-indexing later.)
                        idxs_np = np.arange(labels_np.shape[0])
                    if hasattr(base_ds, "annotations"):
                        # Your CellPaintingDataset keeps metadata in a DataFrame called .annotations
                        names = [
                            str(base_ds.annotations.iloc[int(i)]["Metadata_Sample_ID"])
                            for i in idxs_np
                        ]
                    else:
                        # If you don't have a DataFrame, synthesize a name from fields you already have
                        names = [f"{b}_{p}_{w}" for b, p, w in zip(batch_ids_np, plate_ids_np, well_ids_np)]
                    emb_list.append(emb.detach().cpu().numpy())
                    name_list.extend(names)
                    id_list.extend(idxs_np.tolist())
                    ytrue_list.extend(labels_np.tolist())
                    ypred_list.extend(preds_np.tolist())

                if (phase == "Test") or self.config.get("records", False) or self.config.get("gradcam", False):
                    record_array.append(batch_records)

                if self.config.get("gradcam", False):
                    B = labels_t.size(0)
                    for j in range(B):
                        selector.add({
                            "ds_idx": int(idxs_np[j]),
                            "true":   int(labels_np[j]),
                            "pred":   int(preds_np[j]),
                            "conf":   float(conf_class[j]),
                            "name":   f"{batch_ids_np[j]}_{plate_ids_np[j]}_{well_ids_np[j]}",     
                                   })

        if self.config.get("return_channelwise_embeddings", False) and len(emb_list) > 0 and save_embeddings:
            out_dir = os.path.join(self.ckpt.base_dir, "channelwise_embeddings")
            os.makedirs(out_dir, exist_ok=True)
            payload = {
                "ids":     np.asarray(id_list, dtype=np.int64),
                "names":   np.asarray(name_list, dtype=object),
                "y_true":  np.asarray(ytrue_list, dtype=np.int64),
                "y_pred":  np.asarray(ypred_list, dtype=np.int64),
                "emb":  np.concatenate(emb_list, axis=0),   # [N, D]
            }
            out_path = os.path.join(out_dir, f"{phase.lower()}_channelwise_embeddings.npz")
            np.savez_compressed(out_path, **payload)
            print(f"Saved channelwise embeddings to {out_path}")
        
        acc = correct / max(1, total)
        avg_loss = loss_total / max(1, len(loader))
        print(f"Epoch [{epoch}/{num_epochs}], {phase} Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

        metric = avg_loss if self.early_stopping_metric == "Validation Loss" else acc
        improved = False
        if self.best_metric is None or (metric < self.best_metric if "Loss" in self.early_stopping_metric else metric > self.best_metric):
            self.best_metric = metric
            self.epochs_since_improvement = 0
            improved = True
        else:
            self.epochs_since_improvement += 1

        wandb.log({"Epoch": epoch, f"{phase} Accuracy": acc, f"{phase} Loss": avg_loss})
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        wandb.log({f"{phase}/report": report})

        
        if improved and phase == "Validation" and epoch % 5 != 0 and epoch > 2:
            self.ckpt.save_checkpoint(self.model, epoch, self.config["architecture"], self.optimizer, improve=True)
            metric_key = self.early_stopping_metric.lower().replace(" ", "_")
            wandb.run.summary[f"best_{metric_key}"] = float(metric)
            wandb.run.summary["best_epoch"] = int(epoch)
            

        if ((self.config.get("records", False) and improved) or (phase == "Test")) and len(record_array) > 0:
            record_array = np.concatenate(record_array, axis=0)
            self.ckpt.save_records(record_array, phase)

        if phase == "Test":
            self.ckpt.save_checkpoint_test(self.model, self.config["architecture"], loss=avg_loss)
            print(f"Test Loss: {avg_loss:.4f}")

        if self.config.get("gradcam", False):
            
            selected = selector.selected()
            if selected:
                gcv = GradCAMVisualizer(
                    model=self.model,
                    device=self.device,
                    save_dir=os.path.join(self.ckpt.base_dir, "gradcam"),
                    log_to_wandb=True,
                    channel_names=["Mito", "AGP", "RNA", "ER", "DNA"],
                )
                ds = loader.dataset
                base_ds = ds.base if hasattr(ds, "base") else ds

                for rec in selected:
                    img, lab, *_ = base_ds[rec["ds_idx"]]        # [C,H,W]
                    # Prefer canonical sample name if available
                    name = rec["name"]
                    if hasattr(base_ds, "annotations"):
                        name = str(base_ds.annotations.iloc[rec["ds_idx"]]["Metadata_Sample_ID"])

                    gcv.visualize(
                        input_tensor=img,
                        true_label=rec["true"],
                        pred_label=rec["pred"],
                        original_image_name=name,
                        target_class="true",
                        mode=self.config.get("mode", "combined"),
                    )



        if self.epochs_since_improvement >= self.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch} — no improvement in {self.early_stopping_patience} epochs.")
            return True

        return False


# main entry point
def main():
    args = parse_args()
    config = args_to_config(args)
    config = validate_config(config)

    run_name = None
    wandb_id = None
    start_epoch = 0
    checkpoint = None
    resume_weights = bool(config.get("resume", False)) and not bool(config.get("embedding_mode", False))
    if config.get("resume", False):
        checkpoint_path = config['checkpoint']
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        path_split = checkpoint_path.split("/")
        # Support both flat (checkpoints/run_id/model.pth) and grouped
        # (checkpoints/group/run_id/model.pth) checkpoint layouts.
        run_name_wandb_id = path_split[-2]
        run_name, wandb_id = run_name_wandb_id.rsplit("_", 1)
        # Root dir for SplitManager: grandparent of the checkpoint file
        checkpoint_root_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        if resume_weights:
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch'] + 1
            print(f"🔄 Resuming from checkpoint: {checkpoint_path}, run_name: {run_name}, wandb_id: {wandb_id}")
        else:
            print(f"Resuming an existing wandb run {run_name} without loading weights (-> TVN mode).")


    wandb.init(entity="pilsvera", project="mm", config=config, name=run_name, id=wandb_id, resume="allow")
    # Force CLI config to take precedence over resumed wandb config
    wandb.config.update(config, allow_val_change=True)
    run_name = wandb.run.name
    wandb_id = wandb.run.id
    assert wandb.run is not None, "wandb.run not initialized"
    assert wandb.run.id and wandb.run.name, "wandb id/name missing"
    print(f"Architecture: {wandb.config.get('architecture', 'unknown')}")

    # When resuming from a grouped checkpoint layout (e.g. checkpoints/group/run_id/),
    # use the same parent dir so outputs land next to the original checkpoint.
    output_dir = checkpoint_root_dir if config.get("resume", False) else "checkpoints"
    if config.get("embedding_mode", False):
        ckpt = CheckpointManager(run_name, wandb_id, output_dir=output_dir, subfolder="linear_probing")
        ckpt._make_run_name_dir()
    else:
        ckpt = CheckpointManager(run_name, wandb_id, output_dir=output_dir)
    if not config.get("resume", False):
        ckpt._make_run_name_dir()   
    ckpt.save_config(wandb.config)

    builder = BuildComponents(wandb.config)
    if config.get("synthetic", False):
        train_df = pd.DataFrame({"idx": range(10000)})
        eval_df  = pd.DataFrame({"idx": range(1000)})
        test_df  = pd.DataFrame({"idx": range(1000)})
    else:
        df = builder.dataset()

    tvn_infer_only = config.get("embedding_mode", False) and config.get("inference", False)
    if tvn_infer_only:
        # No splits; build only the test loader from the full df
        train_df = eval_df = None
        test_df  = df.reset_index(drop=True)

    else:
        root_dir = checkpoint_root_dir if config.get("resume", False) else "checkpoints"
        split_manager = SplitManager(wandb.config, df, run_name, wandb_id, root_dir=root_dir)

        if config.get("resume", False):
            train_df, eval_df, test_df = split_manager.load_split_keys()
            print(f"Loading existing data splits from {split_manager.path}")
            strategy = split_manager.load_split_strategy()
            if strategy != config["splits"]:
                raise ValueError(f"Splits strategy mismatch: {strategy} != {config['splits']}")

        else:
            train_df, eval_df, test_df = split_manager.create_splits()
            split_manager.save_split_keys(train_df,  eval_df, test_df)

    train_loader, eval_loader, test_loader = builder.dataloaders(train_df, eval_df, test_df)
    model = builder.model()
    optimizer, scheduler = builder.optimizer(model)
    criterion = builder.criterion()
    
    if resume_weights:
        sd_old = checkpoint['model_state_dict']
        sd_new = model.state_dict()
        arch = config.get("architecture", "ResNet50_Modified")

        skip_optimizer_load = False
        if arch == "OpenPhenomMAE":
            # For MAE: encoder comes from HuggingFace pretrained, only load classifier
            migrated = {}
            for k, v in sd_old.items():
                if k.startswith("classifier."):
                    migrated[k] = v
            # Check classifier shape match
            if "classifier.weight" in migrated and "classifier.weight" in sd_new:
                if sd_new["classifier.weight"].shape != migrated["classifier.weight"].shape:
                    print("⚠️ Classifier shape mismatch; keeping randomly initialized classifier.")
                    print("⚠️ Skipping optimizer state load due to shape mismatch.")
                    migrated = {}
                    skip_optimizer_load = True
            if migrated:
                incomp = model.load_state_dict(migrated, strict=False)
                print(f"Loaded MAE classifier: {list(migrated.keys())}")
            else:
                print("No classifier weights loaded (using fresh initialization)")
                incomp = type('obj', (object,), {'missing_keys': [], 'unexpected_keys': []})()
        else:
            # ResNet loading logic
            migrated = {}
            for k, v in sd_old.items():
                if k.startswith("resnet.fc."):
                    migrated[k.replace("resnet.fc.", "fc.")] = v
                else:
                    migrated[k] = v

            if "fc.weight" in migrated and "fc.weight" in sd_new and sd_new["fc.weight"].shape != migrated["fc.weight"].shape:
                print("⚠️ Classifier shape mismatch; keeping randomly initialized fc.")
                migrated.pop("fc.weight", None); migrated.pop("fc.bias", None)

            incomp = model.load_state_dict(migrated, strict=False)
            print("Loaded with:", "missing:", incomp.missing_keys, "unexpected:", incomp.unexpected_keys)

        if not skip_optimizer_load:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"Skipping optimizer state load: {e}")

        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if config.get("resume", False) and not resume_weights:
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint.get('epoch', -1) + 1
        # sanity: same model type & shapes
        incomp = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("Loaded TVN linear head; missing:", incomp.missing_keys, "unexpected:", incomp.unexpected_keys)
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            print(f"Skipping optimizer load: {e}")
            print(f"Resuming W&B run only (embedding_mode=True).")

   
    trainer = Trainer(model, optimizer, criterion, builder.device, config, run_name, wandb_id, ckpt)
    trainer.scheduler = scheduler
    num_epochs = config["epochs"]

    if not config["inference"] and not config.get("return_channelwise_embeddings", False):

        for epoch in tqdm(range(start_epoch, num_epochs)):
            trainer.training(train_loader, epoch, num_epochs)
            if config.get("lr_scheduler") == "auto":       # using schedulefree
                trainer.optimizer.eval()                   # schedulefree requirement
                trainer.model.train()                      # BN updates running stats in train mode
                with torch.no_grad():
                    for x, *_ in itertools.islice(train_loader, min(50, len(train_loader))):
                        x = x.to(trainer.device, non_blocking=True)
                        _ = trainer.model(x)               # forward only; no loss/step
                trainer.model.eval()          
            if trainer.evaluate(eval_loader, epoch, num_epochs, phase="Validation", gradcam=config.get("gradcam", False), save_embeddings=False):
                break
        print("Training complete.")
    
    elif config.get("return_channelwise_embeddings", False):
        print("Dumping channelwise embeddings for train/val splits")
        trainer.evaluate(train_loader, start_epoch, num_epochs, phase="Train",
                         gradcam=False, save_embeddings=True)
        trainer.evaluate(eval_loader,  start_epoch, num_epochs, phase="Validation",
                         gradcam=False, save_embeddings=True)
        trainer.evaluate(test_loader,  start_epoch, num_epochs, phase="Test",
                         gradcam=False, save_embeddings=True)

    else: 
        start_epoch = num_epochs
        print("Running inference on test set")
        if config.get("lr_scheduler") == "auto" and (train_loader is not None):
            trainer.optimizer.eval()
            trainer.model.train()
            with torch.no_grad():
                for x, *_ in itertools.islice(train_loader, min(50, len(train_loader))):
                    x = x.to(trainer.device, non_blocking=True)
                    _ = trainer.model(x)
            trainer.model.eval()
        trainer.evaluate(test_loader, start_epoch, num_epochs, phase="Test", gradcam=config.get("gradcam", False), save_embeddings=False)
            

       

    if config["debug_mode"]:
        shutil.rmtree(os.path.join("checkpoints", f"{run_name}_{wandb_id}"))
    wandb.finish()


if __name__ == "__main__":
    print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    print("Current device idx:", torch.cuda.current_device() if torch.cuda.is_available() else None)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("cuDNN available:", torch.backends.cudnn.is_available())
    if not torch.cuda.is_available():
        print("No CUDA available")
    else:
        main()


    

