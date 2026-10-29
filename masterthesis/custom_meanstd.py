
import enum
import json
import os
import numpy as np
import tifffile as tiff
import pandas as pd
import torch
from tqdm import tqdm


class MeanStd(enum.Enum):
    IMAGENET = ([0.485, 0.456, 0.406, 0.449, 0.449], [0.229, 0.224, 0.225, 0.226, 0.226])
    CELLPAINTING_1 = ([0.1361, 0.1512, 0.1220, 0.1317, 0.1048], [0.1342, 0.1477, 0.0957, 0.1299, 0.0876])
    CELLPAINTING_1_w_NEG = ([0.15297285, 0.16582283, 0.13343945, 0.13987823, 0.09718288], [0.1378316,  0.14931116, 0.09570032, 0.12717961, 0.09052722])
    CELLPAINTING_3 = ([0.08791038, 0.08602992, 0.10520343, 0.08057716, 0.06542926], [0.09324875, 0.0899062,  0.08386253, 0.06686723, 0.07686927])
    CELLPAINTING_3_w_NEG = ([0.09862981, 0.09573705, 0.11796389, 0.08198917, 0.07194847], [0.09485177, 0.09217019, 0.0849737,  0.06675208, 0.08214165])
    MAE = (None, None) 
    CUSTOM = ([0.5]*5, [0.5]*5)

    @property
    def mean(self):
        return torch.tensor(self.value[0]) if self.value[0] is not None else None

    @property
    def std(self):
        return torch.tensor(self.value[1]) if self.value[1] is not None else None
    


class CustomMeanAndStd:
    def __init__(self, img_dir: str, index_path: str):
        """
        Args:
            img_dir (str): Directory where images are stored.
            index_path (str): Path to a CSV listing image sample IDs.
        """
        self.img_dir = img_dir
        self.index_path = index_path
        self.mean = None
        self.std = None

    def _find_image_path(self, sample_id: str) -> str:
        """
        Constructs the image filename based on the sample ID.
        """
        return os.path.join(self.img_dir, f"{sample_id}.jpg")

    def calculate(self):
        """
        Calculates the mean of means and mean of stds over all images listed in index.
        """
        index_df = pd.read_parquet(self.index_path)
        sample_ids = index_df['Metadata_Sample_ID'].values

        all_means = []
        all_stds = []

        for sample_id in tqdm(sample_ids, desc="Processing images"):
            img_path = self._find_image_path(sample_id)
            if not os.path.exists(img_path):
                print(f"❌ Warning: Image {img_path} not found, skipping.")
                continue

            img = tiff.imread(img_path).astype(np.float32) / 255.0  # Normalize
            mean = img.mean(axis=(0, 1))  # mean per channel
            std = img.std(axis=(0, 1))    # std per channel
            all_means.append(mean)
            all_stds.append(std)

        if len(all_means) == 0:
            raise ValueError("No valid images found to compute mean and std.")

        all_means = np.stack(all_means)
        all_stds = np.stack(all_stds)

        self.mean = all_means.mean(axis=0)
        self.std = all_stds.mean(axis=0)

        print("Finished computing mean and std.")
        print(f"Mean: {self.mean}")
        print(f"Std: {self.std}")

    def get(self):
        """
        Returns the computed mean and std.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Mean and std have not been calculated yet. Call calculate() first.")
        return self.mean, self.std

    def save(self, output_path: str):
        """
        Saves the mean and std into a JSON file.

        Args:
            output_path (str): Path to save the JSON file.
        """
        if self.mean is None or self.std is None:
            raise ValueError("No mean and std to save. Call calculate() first.")

        data = {
            "mean": self.mean.tolist(),  # Convert numpy array to Python list
            "std": self.std.tolist()
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
        
        print(f"✅ Mean and std saved to {output_path}")


