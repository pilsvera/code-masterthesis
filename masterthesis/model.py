
import os
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import torch.nn as nn
from huggingface_mae import MAEModel


class ResNet50_Modified(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze_encoder=False, return_channelwise_embeddings=False, embedding_mode = "joint"):
        super().__init__()
        self.return_channelwise_embeddings = return_channelwise_embeddings

        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.embedding_mode = embedding_mode
        assert self.embedding_mode in ["joint", "channelwise"], "embedding_mode must be 'joint' or 'channelwise'"

        # adapt to 5 channels
        old = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(5, old.out_channels, old.kernel_size, old.stride, old.padding, bias=(old.bias is not None))

        self.embed_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.fc = nn.Linear(self.embed_dim, num_classes)

        if freeze_encoder:
            self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.resnet.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def _channelwise_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """ Return [B, C, D] embeddings by zeroing all but one input channel per pass. """
        B, C, H, W = x.shape
        embs = []
        for c in range(C):
            xc = torch.zeros_like(x)
            xc[:, c:c+1] = x[:, c:c+1]
            zc = self.resnet(xc)          # [B, D] because fc=Identity
            embs.append(zc)
        return torch.stack(embs, dim=1)    # [B, C, D] 

    def forward(self, x):
        """Default returns logits [B, num_classes]. If return_channelwise=True,
        return (logits, ch_emb [B,C,D])."""
        z = self.resnet(x)         # [B, D]
        logits = self.fc(z)        # [B, num_classes]
        if self.return_channelwise_embeddings and self.embedding_mode == "channelwise":
            ch = self._channelwise_embeddings(x)  # [B, C, D]
            return logits, ch
        if self.return_channelwise_embeddings and self.embedding_mode == "joint":
            return logits, z
        return logits





class ResNet50SingleChannel(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50SingleChannel, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

        # Modify first layer to accept single-channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove classification layer, we only need feature extraction
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove last FC layer

    def forward(self, x):
        x = self.feature_extractor(x)  # Get feature map (Batch, 2048, 1, 1)
        return x.view(x.shape[0], -1)  # Flatten (Batch, 2048)

class MultiChannelResNet50(nn.Module):
    def __init__(self, num_classes, pretrained =True):
        super(MultiChannelResNet50, self).__init__()
        
        # Create 5 independent ResNet50 feature extractors
        self.resnets = nn.ModuleList([ResNet50SingleChannel(pretrained=pretrained) for _ in range(5)])

        # Final classification layer (5 * 2048 features → num_classes)
        self.fc = nn.Linear(5 * 2048, num_classes)

    def forward(self, x):
        # Split input into 5 separate channels (B, 5, H, W) → 5 tensors (B, 1, H, W)
        channel_features = [resnet(x[:, i:i+1, :, :]) for i, resnet in enumerate(self.resnets)]

        # Concatenate features from all channels (B, 5 * 2048)
        combined_features = torch.cat(channel_features, dim=1)

        output = self.fc(combined_features)
        return output








class OpenPhenomMAE(nn.Module):
    def __init__(self, 
                 model_path="recursionpharma/OpenPhenom", 
                 return_channelwise_embeddings=True, 
                 num_classes=None,
                 freeze_encoder=False):
        """
        Args:
            model_path (str): Hugging Face model identifier or path.
            return_channelwise_embeddings (bool): If True, returns per-channel embeddings.
            num_classes (int or None): If provided, adds a classification head.
            freeze_encoder (bool): If True, freezes MAE backbone.
        """
        super(OpenPhenomMAE, self).__init__()

        self.return_channelwise_embeddings = return_channelwise_embeddings

        self.encoder = MAEModel.from_pretrained(model_path)
        self.encoder.return_channelwise_embeddings = return_channelwise_embeddings

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.encoder.parameters():
                param.requires_grad = True

        if num_classes is not None:
            dummy = torch.randn(1, 5, 256, 256)
            with torch.no_grad():
                embeddings = self.encoder.predict(dummy)
                print(f"Shape of embeddings: {embeddings.shape}")
                emb_dim = embeddings.shape[-1]
            self.classifier = nn.Linear(emb_dim, num_classes)
        else:
            self.classifier = None



    def forward(self, x):
        """
        Args:
            x (Tensor): Shape (B, C, H, W)

        Returns:
            Either:
                - embeddings (if classifier is None)
                - logits (if classifier is used and return_channelwise_embeddings is False)
                - (logits, embeddings) tuple (if classifier is used and return_channelwise_embeddings is True)
        """
        with torch.no_grad() if not self.training or not any(p.requires_grad for p in self.encoder.parameters()) else torch.enable_grad():
            embeddings = self.encoder.predict(x)

        if self.classifier:
            logits = self.classifier(embeddings)
            if self.return_channelwise_embeddings:
                return logits, embeddings
            return logits
        return embeddings

class FindLayer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.last_conv_layer = self.find_last_conv_layer(self.model)

    def find_last_conv_layer(self, module: nn.Module) -> nn.Module:
        """
        Recursively find the last nn.Conv2d layer in the model.
        
        :param module: A PyTorch module (e.g., the model or submodule).
        :return: The last nn.Conv2d layer found.
        """
        last_conv = None
        for child in module.children():
            found = self.find_last_conv_layer(child)
            if found:
                last_conv = found
            elif isinstance(child, nn.Conv2d):
                last_conv = child
        return last_conv

class LinearOnEmbeddings(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()

        self.net = nn.Linear(in_dim, num_classes)
    def forward(self, x):  # x: [B, in_dim]
        return self.net(x)


if __name__ == "__main__":
    model = OpenPhenomMAE(model_path="recursionpharma/OpenPhenom", num_classes=3, return_channelwise_embeddings=True)


    model.eval()

    # Step 2: Create a dummy 5-channel input image (batch size = 1)
    dummy_input = torch.randn(1, 5, 256, 256)

    # Step 3: Get embeddings
    with torch.no_grad():
        embeddings = model(dummy_input)

    # Step 4: Print shape and sample values
    print(f"✅ Embedding shape: {embeddings.shape}")