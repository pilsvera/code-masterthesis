import os
from typing import Tuple
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
import albumentations as A
import tifffile as tiff
from abc import ABC, abstractmethod
import json
from custom_meanstd import MeanStd
from mm_logging import get_git_hash

class BuildIndex:
    def __init__(self, source_types: list[str], index_full_path: str, named_path: str = "pos_ctrl_name.csv"): #sources all, 1-11
        '''
        Builds a DataFrame from a Parquet index, filtered by source.

        Args:
            index_full_path (str): Path to the Parquet file.
            source_types (SourceType | list[str]): Single source or list of sources.
        '''
        self.source_types = source_types
        self.index_full_path = index_full_path
        self.index_full_path == "all_sources.pq"
        self.is_neg = "_w_neg"
        self.named_path = named_path
        self.dataset, self.size = self.build_index()

    def _extract_source(self) -> pd.DataFrame:
        df = pd.read_parquet(self.index_full_path)
        if self.source_types == 'all':
            return df
        elif isinstance(self.source_types, list):
            return df[df['Metadata_Source'].isin(self.source_types)]
        else:
            return df[df['Metadata_Source'] == self.source_types]

            
    def _merge_name(self, df:pd.DataFrame):
        '''
        Merges the index with a positive control annotation file.
        '''
        named_df = pd.read_csv(self.named_path, sep='\t')
        df = df.merge(named_df, left_on='Metadata_JCP2022', right_on='jcp2022_id', how='left')
        print(df.columns)
        return df
    
    def _encode_categorical_to_numerical(self, df: pd.DataFrame, column) -> pd.DataFrame:
        '''
        Encodes categorical columns to numerical values.
        '''
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in DataFrame.")
        df[column + '_str'] = df[column].copy()
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        self._save_encoding(le, column) 
        return df

    def _save_encoding(self, encoder: LabelEncoder, column: str):
        encoding_dict = {
            str(k): int(v)  
            for k, v in zip(encoder.classes_, encoder.transform(encoder.classes_))
        }
        src = "_".join(self.source_types) if isinstance(self.source_types, list) else str(self.source_types)
        filename = f"{src}_encoding{self.is_neg}_{column}.json"
        with open(filename, "w") as f:
            json.dump(encoding_dict, f)
        print(f"Encoding saved to {filename}")


    def _save_df(self, df: pd.DataFrame):
        '''
        Saves the DataFrame to a pq file.
        '''
        if isinstance(self.source_types, list):
            filename = f"{'_'.join(self.source_types)}{self.is_neg}.pq"
        else:
            filename = f"{self.source_types}{self.is_neg}.pq"
        df.to_parquet(filename, index=False)
        print(f"DataFrame saved to {filename}")


    def build_index(self) -> tuple[pd.DataFrame, int]:
        df = self._extract_source()
        df = self._merge_name(df)
        df = self._encode_categorical_to_numerical(df, 'pert_iname')
        df = self._encode_categorical_to_numerical(df, 'Metadata_Batch')   
        df = self._encode_categorical_to_numerical(df, 'Metadata_Plate')
        df = self._encode_categorical_to_numerical(df, 'Metadata_Well')
        self._save_df(df)
        return df, len(df)


class WithIndex(Dataset):
    """Wraps any dataset so __getitem__ returns (..., idx)"""
    def __init__(self, base: Dataset):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        out = self.base[idx]
        # Ensure tuple; append the index
        if isinstance(out, tuple):
            return (*out, idx)
        return out, idx

    
class SyntheticDataset(Dataset):
    def __init__(self, num_samples: int, num_classes: int, image_size=(5, 224, 224), device: str = "cpu"):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.C, self.H, self.W = image_size
        self.device = device  # keep CPU; DataLoader will move to GPU later

    def __len__(self):
        return self.num_samples


    def __getitem__(self, idx):
        x = torch.randn(self.C, self.H, self.W, dtype=torch.float32)   # on CPU
        y = torch.randint(0, self.num_classes, (1,), dtype=torch.long).item()
        # Stub metadata to match your Trainer.evaluate signature
        is_ctrl = False
        plate_id = "syn_plate"
        well = "A01"
        batch_id = 0
        return x, y, is_ctrl, plate_id, well, batch_id

    

   


class CellPaintingDataset(Dataset):
    def __init__(self, img_dir: str, df: pd.DataFrame, transform=None):
        '''
        Dataset class for Cell Painting images.

        Args:
            img_dir (str): Directory containing image files.
            df (pd.DataFrame): Metadata DataFrame.
            transform (callable, optional): Augmentation pipeline.
        '''
        self.img_dir = img_dir
        self.annotations = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_name = os.path.join(self.img_dir, row['Metadata_Sample_ID'] + '.jpg')

        if not os.path.exists(img_name):
            raise FileNotFoundError(f'Image {img_name} not found!')

        img = tiff.imread(img_name).astype(np.float32) / 255.0

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img)
            img = torch.moveaxis(img, 2, 0)

        label = torch.tensor(row['pert_iname'], dtype=torch.long)
        is_ctrl = torch.tensor(1 if row['pert_iname_str'] == 'DMSO' else 0)        
        plate_id = row['Metadata_Plate']
        well = row['Metadata_Well']
        batch_id = row['Metadata_Batch']

        return img, label, is_ctrl, plate_id, well, batch_id


class MiniDataset(CellPaintingDataset):
    def __init__(self, img_dir: str, df: pd.DataFrame, transform=None, random_state: int = 42, max_samples: int = 5000):
        '''
        Mini dataset for quick testing/debugging.
        '''
        mini_df = df.sample(n=min(len(df), max_samples), random_state=random_state).reset_index(drop=True)
        super().__init__(img_dir, mini_df, transform=transform)

class TVNEmbeddingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, embedding_cols: list[int], label_col: str = "pert_iname"):
        """
        Args:
            df (pd.DataFrame): The DataFrame containing metadata and embeddings.
            embedding_cols (list[int]): List of column names for the embeddings.
            label_col (str): Name of the column containing the classification label.
        """
        self.df = df.reset_index(drop=True)
        self.embedding_cols = embedding_cols
        self.label_col = label_col
        

    def __len__(self):
        return len(self.df)

   
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        embedding = torch.tensor(row[self.embedding_cols].values.astype(np.float32))
        label = torch.tensor(row[self.label_col], dtype=torch.long)
        is_ctrl = torch.tensor(bool(row.get("is_control", False)))
        plate_id = row.get("plate", "unknown")
        well = row.get("well", "unknown")
        batch_id = row.get("batch", "unknown")
        return embedding, label, is_ctrl, plate_id, well, batch_id




class FlexibleUndersamplingStrategy:
    def __init__(self, per_class_count: int = None, class_specific_counts: dict = None):
        """
        Initialize the strategy.

        Args:
            per_class_count (int): If set, this number of samples is used for all classes.
            class_specific_counts (dict): Optionally specify per-class sample counts.
        """
        self.per_class_count = per_class_count
        self.class_specific_counts = class_specific_counts

    def apply(self, dataloader):
        """
        Apply undersampling to the dataloader's dataset (expects a `.data` DataFrame).

        Args:
            dataloader: PyTorch DataLoader with dataset that has a `.data` DataFrame.
        
        Returns:
            Updated DataLoader with downsampled `.dataset.data`.
        """
        df = dataloader.dataset.data
        label_col = df.columns[-1]  # assumes label is last column (adjust if needed)

        resampled_data = []

        for label, group in df.groupby(label_col):
            if self.class_specific_counts and label in self.class_specific_counts:
                n = self.class_specific_counts[label]
            elif self.per_class_count:
                n = min(self.per_class_count, len(group))
            else:
                raise ValueError("Either `per_class_count` or `class_specific_counts` must be set.")

            sampled_group = group.sample(n=n, replace=False, random_state=42)
            resampled_data.append(sampled_group)

        resampled_df = pd.concat(resampled_data).reset_index(drop=True)
        dataloader.dataset.data = resampled_df
        return dataloader



class FiveChannelAlbumentations:
    def __init__(self, config: dict, mode: str = 'train'):
        '''
        Albumentations-based augmentation pipeline for 5-channel images.
        '''
        self.config = config
        self.mode = mode
        
        # Resolve normalization explicitly and ALWAYS log it — `mean_std` must be
        # present in this dict (the caller is responsible for propagating it from
        # the top-level config). If it is missing we fall back to CELLPAINTING_1,
        # but say so loudly rather than silently.
        _ms_key = config.get('mean_std', 'CELLPAINTING_1')
        _fallback = 'mean_std' not in config
        self.mean_std = MeanStd[_ms_key]
        self.mean = self.mean_std.mean
        self.std = self.mean_std.std
        print(f"[FiveChannelAlbumentations] mode={mode}: normalization mean_std={_ms_key}"
              + ("  (DEFAULT FALLBACK — 'mean_std' not in config dict)" if _fallback else ""))
        self.pipeline = self._build_pipeline()
        print(f"Built albumentations pipeline {mode}:", self.pipeline)

    def _build_pipeline(self):
        """Constructs the albumentations pipeline from the config."""
        cfg = self.config
        aug_list = []

        if self.mode == 'train':
            if cfg.get('RandomResizedCrop'):
                rrc = cfg['RandomResizedCrop']
                size = (cfg['resize'][0], cfg['resize'][1])
                aug_list.append(A.RandomResizedCrop(
                    size=size,
                    scale=rrc.get('scale', (0.5, 1.0)),
                    ratio=rrc.get('ratio', (1, 1)),
                    p=rrc.get('prob', 0.5),
                ))
            if cfg.get('horizontal_flip_prob', False):
                aug_list.append(A.HorizontalFlip(p=cfg['horizontal_flip_prob']))
            if cfg.get('vertical_flip_prob', False):
                aug_list.append(A.VerticalFlip(p=cfg['vertical_flip_prob']))
            if cfg.get('rotation_prob', 0):
                aug_list.append(A.RandomRotate90(p=cfg['rotation_prob']))
            if cfg.get('noise_std', False):
                std = cfg.get('noise_std', 0.1)
                aug_list.append(A.GaussNoise(std_range=(std, std), p=cfg.get('noise_prob', 0.5)))
            if cfg.get('blur_prob', 0):
                aug_list.append(A.GaussianBlur(blur_limit=(3, 7), p=cfg['blur_prob']))
            if cfg.get('brightness_contrast_prob', False):
                aug_list.append(A.RandomBrightnessContrast(p=cfg.get('brightness_contrast_prob', 0.5)))
            if cfg.get('coarse_dropout_prob', 0):
                aug_list.append(A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(0.05, 0.15),
                    hole_width_range=(0.05, 0.15),
                    fill=0,
                    p=cfg['coarse_dropout_prob'],
                ))
            if cfg.get('plasma_shadow'):
                ps = cfg['plasma_shadow']
                aug_list.append(A.PlasmaShadow(
                    shadow_intensity_range=ps.get('shadow_intensity_range', (0.3, 0.7)),
                    roughness=ps.get('roughness', 3.0),
                    p=ps.get('prob', 0.5),
                ))
        if cfg.get('resize', False):
            aug_list.append(A.Resize(cfg['resize'][0], cfg['resize'][1]))

        return A.Compose(aug_list)

    def __call__(self, img):
        """
        Applies the augmentation pipeline to a 5-channel image.

        Args:
            img (torch.Tensor or np.ndarray): Shape (C, H, W) or (H, W, C)

        Returns:
            torch.Tensor: Augmented and normalized image, shape (C, H, W)
        """
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
        augmented = self.pipeline(image=img)['image']
        tensor_img = torch.tensor(augmented).permute(2, 0, 1).float()
        if self.mean is not None and self.std is not None:
            tensor_img = (tensor_img - self.mean[:, None, None]) / self.std[:, None, None]
        return tensor_img # (C, H, W)


class Splits(ABC):    
    @abstractmethod    
    def get_train(self):        
        pass        
    
    @abstractmethod    
    def get_eval(self):        
        pass
    
    @abstractmethod    
    def get_test(self):        
        pass

    def _split_keys(self, df: pd.DataFrame, group_column: str, eval_size: float, test_size: float, random_state: int):        
        gss = GroupShuffleSplit(n_splits=1, test_size=eval_size + test_size, random_state=random_state)        
        groups = df[group_column]            
        # First split: train vs temp (eval + test)        
        train_idx, temp_idx = next(gss.split(df, groups=groups))        
        train_df = df.iloc[train_idx].reset_index(drop=True)        
        temp_df = df.iloc[temp_idx].reset_index(drop=True)
        # Second split: eval vs test from temp        
        gss2 = GroupShuffleSplit(n_splits=1, test_size=test_size / (eval_size + test_size), random_state=random_state)        
        eval_idx, test_idx = next(gss2.split(temp_df, groups=temp_df[group_column]))        
        eval_df = temp_df.iloc[eval_idx].reset_index(drop=True)        
        test_df = temp_df.iloc[test_idx].reset_index(drop=True)
        self._print_split_info(            
            train_df,            
            eval_df,           
            test_df            
             )
        return train_df, eval_df, test_df

    def _print_split_info(self, train_keys, eval_keys, test_keys):        
        total_keys = len(train_keys) + len(eval_keys) + len(test_keys)        
        print("Train keys: %.2f | Eval keys: %.2f | Test keys: %.2f" % (            
            len(train_keys) / total_keys,            
            len(eval_keys) / total_keys,            
            len(test_keys) / total_keys        
            ))


class BatchwiseSplits(Splits):    
    def __init__(self, df: pd.DataFrame, eval_size: float = 0.2, test_size: float = 0.1, random_state: int = 42):        
        self.df = df        
        self.eval_size = eval_size        
        self.test_size = test_size        
        self.random_state = random_state

        self.train_df, self.eval_df, self.test_df = self._split_keys(            
            df,            
            group_column="Metadata_Batch",            
            eval_size=eval_size,            
            test_size=test_size,            
            random_state=random_state        
            )

    def get_train(self):        
        return self.train_df
    def get_eval(self):        
        return self.eval_df
    def get_test(self):        
        return self.test_df

class PlatewiseSplits(Splits):    
    def __init__(self, df: pd.DataFrame, eval_size: float = 0.2, test_size: float = 0.2, random_state: int = 42):        
        self.df = df        
        self.eval_size = eval_size        
        self.test_size = test_size        
        self.random_state = random_state

        self.train_df, self.eval_df, self.test_df = self._split_keys(            
            df,            
            group_column="Metadata_Plate",            
            eval_size=eval_size,            
            test_size=test_size,            
            random_state=random_state
            )
        
    def get_train(self):        
        return self.train_df
    def get_eval(self):        
        return self.eval_df
    def get_test(self):        
        return self.test_df

class WellwiseSplits(Splits):   
    def __init__(self, df: pd.DataFrame, eval_size: float = 0.2, test_size: float = 0.1, random_state: int = 42):        
        self.df = df.copy()        
        self.eval_size = eval_size        
        self.test_size = test_size        
        self.random_state = random_state

        df["well_key"] = list(zip(df["Metadata_Plate"], df["Metadata_Well"]))        
        self.train_df, self.eval_df, self.test_df = self._split_keys(            
            df,            
            group_column="well_key",            
            eval_size=eval_size,            
            test_size=test_size,            
            random_state=random_state        
            )        
        

    def get_train(self):        
        return self.train_df
    def get_eval(self):        
        return self.eval_df
    def get_test(self):        
        return self.test_df


class RandomSplits(Splits):    
    def __init__(self, df: pd.DataFrame, eval_size: float = 0.2, test_size: float = 0.1, random_state: int = 42):        
        self.df = df        
        self.eval_size = eval_size        
        self.test_size = test_size        
        self.random_state = random_state

        self.train_df, self.eval_df, self.test_df = self._split_keys(            
            df,            
            group_column="Metadata_Sample_ID",            
            eval_size=eval_size,            
            test_size=test_size,            
            random_state=random_state        
            )

    def get_train(self):        
        return self.train_df
    def get_eval(self):        
        return self.eval_df
    def get_test(self):        
        return self.test_df






class SplitManager:
    def __init__(self, config: dict, df: pd.DataFrame, run_name: str, wandb_id: str, root_dir: str = "checkpoints"):
        self.config = config
        self.df = df
        self.split_strategy = config["splits"]
        self.test_size = config.get("test_split", 0.1)
        self.eval_size = config.get("eval_split", 0.2)
        self.random_state = config.get("seed", 42)
        self.root_dir = root_dir
        self.run_name = run_name
        self.wandb_id = wandb_id
        base_path = os.path.join(self.root_dir, f"{self.run_name}_{self.wandb_id}")
        self.path = os.path.join(base_path, "split_keys.json")

    def create_splits(self):
        if self.split_strategy == "plates":
            splitter = PlatewiseSplits(self.df, self.eval_size, self.test_size, self.random_state)
        elif self.split_strategy == "wells":
            splitter = WellwiseSplits(self.df, self.eval_size, self.test_size, self.random_state)
        elif self.split_strategy == "random":
            splitter = RandomSplits(self.df, self.eval_size, self.test_size, self.random_state)
        elif self.split_strategy == "batches":
            splitter = BatchwiseSplits(self.df, self.eval_size, self.test_size, self.random_state)
        else:
            raise ValueError(f"Unknown split strategy: {self.split_strategy}")

        return splitter.get_train(), splitter.get_eval(), splitter.get_test()

    def save_split_keys(self, train_df, eval_df, test_df):
        split_info = {
            "split_strategy": self.split_strategy,
            "random_state": self.random_state,
            "git_commit": get_git_hash(),
            "train_keys": list(train_df["Metadata_Sample_ID"]),
            "eval_keys": list(eval_df["Metadata_Sample_ID"]),
            "test_keys": list(test_df["Metadata_Sample_ID"]),
        }

        with open(self.path, "w") as f:
            json.dump(split_info, f, indent=2)
        print(f"Split keys saved to: {self.path}")

    def load_split_keys(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Split keys file not found at: {self.path}")

        with open(self.path, "r") as f:
            split_info = json.load(f)
        

        def filter_df(keys):
            return self.df[self.df["Metadata_Sample_ID"].isin(keys)].reset_index(drop=True)
        return filter_df(split_info["train_keys"]), filter_df(split_info["eval_keys"]), filter_df(split_info["test_keys"])
    
    def load_split_strategy(self) -> str:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Split keys file not found at: {self.path}")
        with open(self.path, "r") as f:
            split_info = json.load(f)
        return split_info["split_strategy"]




def main():
    df = pd.read_csv('source_1.csv')
    splitter = WellwiseSplits(df, eval_size=0.2, test_size=0.1, random_state=42)
    train_df = splitter.get_train()
    eval_df = splitter.get_eval()
    test_df = splitter.get_test()
    
    print("Train label distribution:")
    print(train_df['pert_iname'].value_counts(normalize=True).head())

    print("\nEval label distribution:")
    print(eval_df['pert_iname'].value_counts(normalize=True).head())

    print("\nTest label distribution:")
    print(test_df['pert_iname'].value_counts(normalize=True).head()) 




if __name__ == '__main__':
    import yaml
    config_path = "masterthesis/config/visualizations.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    alb = FiveChannelAlbumentations(config, mode='train')

