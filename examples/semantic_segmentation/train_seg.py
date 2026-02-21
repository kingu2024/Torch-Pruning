"""
Semantic Segmentation Training Script

Supports two modes:

1. Modular mode -- compose your model from backbone + neck + head:
    python train_seg.py --data-root /data --num-classes 21 \
        --backbone resnet50 --neck fpn --head fcn --pretrained-backbone

2. Preset mode -- use torchvision segmentation models directly:
    python train_seg.py --data-root /data --num-classes 21 \
        --model deeplabv3_resnet50

Dataset structure:
    data_root/
        images/    # original images (.jpg)
        segs/      # segmentation masks (.png), pixel values = class indices
"""

import os
import argparse
import logging
from collections import OrderedDict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    deeplabv3_mobilenet_v3_large,
    fcn_resnet50,
    fcn_resnet101,
    lraspp_mobilenet_v3_large,
)

from losses import MultiSegLoss

# =====================================================
# Preset Models (backward compatibility)
# =====================================================

PRESET_MODELS = {
    "deeplabv3_resnet50": deeplabv3_resnet50,
    "deeplabv3_resnet101": deeplabv3_resnet101,
    "deeplabv3_mobilenet_v3_large": deeplabv3_mobilenet_v3_large,
    "fcn_resnet50": fcn_resnet50,
    "fcn_resnet101": fcn_resnet101,
    "lraspp_mobilenet_v3_large": lraspp_mobilenet_v3_large,
}
MODEL_DICT = PRESET_MODELS  # alias for backward compatibility


# =====================================================
# Backbones
# =====================================================

class ResNetBackbone(nn.Module):
    """ResNet backbone extracting multi-scale features (feat1..feat4)."""

    CHANNELS = {
        "resnet18": [64, 128, 256, 512],
        "resnet34": [64, 128, 256, 512],
        "resnet50": [256, 512, 1024, 2048],
        "resnet101": [256, 512, 1024, 2048],
        "resnet152": [256, 512, 1024, 2048],
    }

    def __init__(self, name="resnet50", pretrained=False):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        resnet = getattr(models, name)(weights=weights)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.out_channels = self.CHANNELS[name]

    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return OrderedDict([("feat1", c1), ("feat2", c2), ("feat3", c3), ("feat4", c4)])


class MobileNetV2Backbone(nn.Module):
    """MobileNetV2 backbone extracting multi-scale features."""

    # (feature_index, output_channels) at stride 4, 8, 16, 32
    STAGES = [(3, 24), (6, 32), (13, 96), (18, 1280)]

    def __init__(self, pretrained=False):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        self.features = models.mobilenet_v2(weights=weights).features
        self.out_channels = [ch for _, ch in self.STAGES]
        self._boundaries = {idx: f"feat{i + 1}" for i, (idx, _) in enumerate(self.STAGES)}

    def forward(self, x):
        out = OrderedDict()
        for i, block in enumerate(self.features):
            x = block(x)
            if i in self._boundaries:
                out[self._boundaries[i]] = x
        return out


class MobileNetV3Backbone(nn.Module):
    """MobileNetV3 backbone extracting multi-scale features."""

    CONFIGS = {
        "mobilenet_v3_large": [(3, 24), (6, 40), (12, 112), (16, 960)],
        "mobilenet_v3_small": [(1, 16), (3, 24), (8, 48), (12, 576)],
    }

    def __init__(self, name="mobilenet_v3_large", pretrained=False):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        self.features = getattr(models, name)(weights=weights).features
        stages = self.CONFIGS[name]
        self.out_channels = [ch for _, ch in stages]
        self._boundaries = {idx: f"feat{i + 1}" for i, (idx, _) in enumerate(stages)}

    def forward(self, x):
        out = OrderedDict()
        for i, block in enumerate(self.features):
            x = block(x)
            if i in self._boundaries:
                out[self._boundaries[i]] = x
        return out


BACKBONE_REGISTRY = {
    "resnet18":           lambda p: ResNetBackbone("resnet18", p),
    "resnet34":           lambda p: ResNetBackbone("resnet34", p),
    "resnet50":           lambda p: ResNetBackbone("resnet50", p),
    "resnet101":          lambda p: ResNetBackbone("resnet101", p),
    "resnet152":          lambda p: ResNetBackbone("resnet152", p),
    "mobilenet_v2":       lambda p: MobileNetV2Backbone(p),
    "mobilenet_v3_large": lambda p: MobileNetV3Backbone("mobilenet_v3_large", p),
    "mobilenet_v3_small": lambda p: MobileNetV3Backbone("mobilenet_v3_small", p),
}


def build_backbone(name, pretrained=False):
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone: {name}. Choose from: {list(BACKBONE_REGISTRY.keys())}")
    return BACKBONE_REGISTRY[name](pretrained)


# =====================================================
# Necks
# =====================================================

class FPNNeck(nn.Module):
    """Feature Pyramid Network -- fuses multi-scale features via top-down pathway."""

    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        for in_ch in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_ch, out_channels, 1))
            self.output_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.out_channels = out_channels

    def forward(self, features):
        feat_list = [features[k] for k in features]
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, feat_list)]
        # Top-down fusion
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="bilinear", align_corners=False
            )
        outs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        # Merge all levels to highest resolution
        target_size = outs[0].shape[2:]
        merged = outs[0]
        for o in outs[1:]:
            merged = merged + F.interpolate(o, size=target_size, mode="bilinear", align_corners=False)
        return merged


class ASPPNeck(nn.Module):
    """Atrous Spatial Pyramid Pooling (DeepLabV3 style).

    Operates on the deepest feature map only.
    """

    def __init__(self, in_channels, out_channels=256, atrous_rates=(6, 12, 18)):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            )
        ]
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            ))
        # Global average pooling branch
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        ))
        self.convs = nn.ModuleList(modules)
        n_branches = len(atrous_rates) + 2
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * n_branches, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.5),
        )
        self.out_channels = out_channels

    def forward(self, features):
        x = list(features.values())[-1] if isinstance(features, dict) else features
        res = []
        for conv in self.convs:
            out = conv(x)
            if out.shape[2:] != x.shape[2:]:
                out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
            res.append(out)
        return self.project(torch.cat(res, dim=1))


class PPMNeck(nn.Module):
    """Pyramid Pooling Module (PSPNet style).

    Operates on the deepest feature map only.
    """

    def __init__(self, in_channels, out_channels=256, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList()
        for size in pool_sizes:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            ))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + out_channels * len(pool_sizes), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.1),
        )
        self.out_channels = out_channels

    def forward(self, features):
        x = list(features.values())[-1] if isinstance(features, dict) else features
        priors = [x]
        for stage in self.stages:
            prior = stage(x)
            prior = F.interpolate(prior, size=x.shape[2:], mode="bilinear", align_corners=False)
            priors.append(prior)
        return self.bottleneck(torch.cat(priors, dim=1))


class IdentityNeck(nn.Module):
    """Pass through the deepest feature map unchanged."""

    def __init__(self, in_channels):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, features):
        return list(features.values())[-1] if isinstance(features, dict) else features


NECK_REGISTRY = ["fpn", "aspp", "ppm", "identity"]


def build_neck(name, backbone_channels, out_channels=256):
    """Build neck module.

    Args:
        name: one of fpn, aspp, ppm, identity
        backbone_channels: list [c1, c2, c3, c4] from backbone
        out_channels: neck output channel dimension
    """
    if name == "fpn":
        return FPNNeck(backbone_channels, out_channels)
    elif name == "aspp":
        return ASPPNeck(backbone_channels[-1], out_channels)
    elif name == "ppm":
        return PPMNeck(backbone_channels[-1], out_channels)
    elif name == "identity":
        return IdentityNeck(backbone_channels[-1])
    else:
        raise ValueError(f"Unknown neck: {name}. Choose from: {NECK_REGISTRY}")


# =====================================================
# Heads
# =====================================================

class FCNHead(nn.Module):
    """FCN-style head: Conv-BN-ReLU-Dropout-Conv."""

    def __init__(self, in_channels, num_classes, mid_channels=256):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(mid_channels, num_classes, 1),
        )

    def forward(self, x):
        return self.block(x)


class SimpleHead(nn.Module):
    """Lightweight 1x1 convolution head."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x):
        return self.conv(x)


HEAD_REGISTRY = ["fcn", "simple"]


def build_head(name, in_channels, num_classes):
    if name == "fcn":
        return FCNHead(in_channels, num_classes)
    elif name == "simple":
        return SimpleHead(in_channels, num_classes)
    else:
        raise ValueError(f"Unknown head: {name}. Choose from: {HEAD_REGISTRY}")


# =====================================================
# Modular Segmentation Model
# =====================================================

class SegModel(nn.Module):
    """Segmentation model composed of backbone + neck + head.

    Always returns a dict: {"out": Tensor, "aux": Tensor (optional, training only)}.
    This makes it compatible with torchvision segmentation model interface.
    """

    def __init__(self, backbone, neck, head, aux_head=None):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.aux_head = aux_head

    def forward(self, x):
        input_shape = x.shape[2:]
        features = self.backbone(x)
        neck_out = self.neck(features)
        out = self.head(neck_out)
        out = F.interpolate(out, size=input_shape, mode="bilinear", align_corners=False)
        result = {"out": out}
        if self.aux_head is not None and self.training:
            feat_values = list(features.values())
            aux_feat = feat_values[-2] if len(feat_values) >= 2 else feat_values[-1]
            aux = self.aux_head(aux_feat)
            aux = F.interpolate(aux, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = aux
        return result


def build_seg_model(backbone_name, neck_name, head_name, num_classes,
                    pretrained_backbone=False, neck_channels=256, aux_loss=False):
    """Build a modular segmentation model from component names."""
    backbone = build_backbone(backbone_name, pretrained=pretrained_backbone)
    neck = build_neck(neck_name, backbone.out_channels, neck_channels)
    head = build_head(head_name, neck.out_channels, num_classes)
    aux_head = None
    if aux_loss:
        aux_in_ch = backbone.out_channels[-2] if len(backbone.out_channels) >= 2 else backbone.out_channels[-1]
        aux_head = FCNHead(aux_in_ch, num_classes)
    return SegModel(backbone, neck, head, aux_head)


def build_model(args):
    """Build model based on args -- dispatches between modular and preset modes.

    Modular mode (--backbone is set):
        Composes model from --backbone + --neck + --head.
    Preset mode (--model is set, no --backbone):
        Loads a torchvision segmentation model by name.
    """
    backbone_name = getattr(args, "backbone", None)
    model_name = getattr(args, "model", None)

    if backbone_name is not None:
        # Modular mode
        neck_name = getattr(args, "neck", "fpn")
        head_name = getattr(args, "head", "fcn")
        neck_channels = getattr(args, "neck_channels", 256)
        pretrained = getattr(args, "pretrained_backbone", False)
        aux_loss = getattr(args, "aux_loss", False)
        return build_seg_model(
            backbone_name, neck_name, head_name, args.num_classes,
            pretrained_backbone=pretrained, neck_channels=neck_channels, aux_loss=aux_loss,
        )
    elif model_name is not None:
        # Preset mode
        if model_name not in PRESET_MODELS:
            raise ValueError(f"Unknown preset model: {model_name}. "
                             f"Choose from: {list(PRESET_MODELS.keys())}")
        if model_name.startswith("lraspp"):
            return PRESET_MODELS[model_name](num_classes=args.num_classes)
        else:
            aux_loss = getattr(args, "aux_loss", False)
            return PRESET_MODELS[model_name](num_classes=args.num_classes, aux_loss=aux_loss)
    else:
        raise ValueError("Specify either --backbone (modular mode) or --model (preset mode)")


# =====================================================
# Dataset
# =====================================================

class SegDataset(Dataset):
    """Custom semantic segmentation dataset.

    Expects:
        root/images/*.jpg   - input images
        root/segs/*.png     - segmentation masks (pixel value = class id)

    The mask and image filenames must match (same stem, different extension).
    """

    def __init__(self, root, image_size=512, is_train=True):
        self.root = root
        self.image_size = image_size
        self.is_train = is_train

        img_dir = os.path.join(root, "images")
        seg_dir = os.path.join(root, "segs")

        img_files = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir)
                     if f.lower().endswith((".jpg", ".jpeg"))}
        seg_files = {os.path.splitext(f)[0]: f for f in os.listdir(seg_dir)
                     if f.lower().endswith(".png")}

        common_stems = sorted(set(img_files.keys()) & set(seg_files.keys()))
        if len(common_stems) == 0:
            raise RuntimeError(
                f"No matching image/mask pairs found in {root}. "
                "Ensure images/*.jpg and segs/*.png share the same filenames."
            )

        self.images = [os.path.join(img_dir, img_files[s]) for s in common_stems]
        self.masks = [os.path.join(seg_dir, seg_files[s]) for s in common_stems]

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])

        if self.is_train:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.5, 2.0), ratio=(0.75, 1.33)
            )
            image = transforms.functional.resized_crop(
                image, i, j, h, w, (self.image_size, self.image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            mask = transforms.functional.resized_crop(
                mask, i, j, h, w, (self.image_size, self.image_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            )
            if torch.rand(1) > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
        else:
            image = transforms.functional.resize(
                image, (self.image_size, self.image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            mask = transforms.functional.resize(
                mask, (self.image_size, self.image_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            )

        image = transforms.functional.to_tensor(image)
        image = self.normalize(image)
        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask


# =====================================================
# Evaluation
# =====================================================

def compute_miou(pred, target, num_classes, ignore_index=255):
    """Compute mean Intersection over Union."""
    pred = pred.view(-1)
    target = target.view(-1)
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    ious = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union == 0:
            continue
        ious.append((intersection / union).item())
    if len(ious) == 0:
        return 0.0
    return np.mean(ious)


@torch.no_grad()
def evaluate(model, val_loader, num_classes, device):
    model.eval()
    total_miou = 0.0
    total_loss = 0.0
    count = 0
    for images, masks in val_loader:
        images, masks = images.to(device), masks.to(device)
        out = model(images)["out"]
        loss = F.cross_entropy(out, masks, ignore_index=255)
        pred = out.argmax(dim=1)
        miou = compute_miou(pred, masks, num_classes)
        total_miou += miou
        total_loss += loss.item()
        count += 1
    return total_miou / max(count, 1), total_loss / max(count, 1)


# =====================================================
# Training
# =====================================================

def get_logger(name, log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def train(args):
    logger = get_logger("train_seg", log_file=os.path.join(args.output_dir, "train.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Dataset
    full_dataset = SegDataset(args.data_root, image_size=args.image_size, is_train=True)
    val_size = max(1, int(len(full_dataset) * args.val_ratio))
    train_size = len(full_dataset) - val_size
    train_dst, val_dst = random_split(full_dataset, [train_size, val_size])
    val_eval_dst = SegDataset(args.data_root, image_size=args.image_size, is_train=False)
    val_eval_subset = torch.utils.data.Subset(val_eval_dst, val_dst.indices)

    train_loader = DataLoader(
        train_dst, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_eval_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    logger.info("Train samples: %d, Val samples: %d", len(train_dst), len(val_eval_subset))

    # Model
    model = build_model(args)

    if args.pretrained_backbone and args.model is not None and args.backbone is None:
        # Preset mode: transfer backbone weights from a pretrained model
        logger.info("Using pretrained backbone weights (preset mode)")
        if args.model.startswith("lraspp"):
            pretrained = PRESET_MODELS[args.model](num_classes=21)
        else:
            pretrained = PRESET_MODELS[args.model](num_classes=21, aux_loss=True)
        model.backbone.load_state_dict(pretrained.backbone.state_dict(), strict=False)
        del pretrained

    if args.restore:
        logger.info("Restoring from %s", args.restore)
        loaded = torch.load(args.restore, map_location="cpu")
        if isinstance(loaded, nn.Module):
            model = loaded
        else:
            model.load_state_dict(loaded, strict=False)

    model = model.to(device)

    if args.backbone is not None:
        logger.info("Model: %s + %s + %s (neck_channels=%d)",
                     args.backbone, args.neck, args.head, args.neck_channels)
    else:
        logger.info("Model: %s (preset)", args.model)

    # Loss function
    criterion = MultiSegLoss(
        num_classes=args.num_classes,
        ce_weight=args.ce_weight,
        edge_weight=args.edge_weight,
        lovasz_weight=args.lovasz_weight,
        ignore_index=255,
    ).to(device)
    logger.info("Loss weights: CE=%.2f, Edge=%.2f, Lovász=%.2f",
                args.ce_weight, args.edge_weight, args.lovasz_weight)

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_miou = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss, loss_dict = criterion(output["out"], masks)
            if "aux" in output and output["aux"] is not None and args.aux_loss:
                aux_ce = F.cross_entropy(output["aux"], masks, ignore_index=255)
                loss = loss + 0.4 * aux_ce
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % args.log_interval == 0:
                loss_info = " ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
                logger.info(
                    "Epoch [%d/%d] Iter [%d/%d] Loss: %.4f (%s) LR: %.6f",
                    epoch + 1, args.epochs, i + 1, len(train_loader),
                    running_loss / (i + 1), loss_info, optimizer.param_groups[0]["lr"],
                )
        scheduler.step()

        # Evaluation
        miou, val_loss = evaluate(model, val_loader, args.num_classes, device)
        logger.info(
            "Epoch [%d/%d] Val mIoU: %.4f Val Loss: %.4f",
            epoch + 1, args.epochs, miou, val_loss,
        )

        if miou > best_miou:
            best_miou = miou
            save_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            logger.info("Saved best model (mIoU=%.4f) to %s", best_miou, save_path)

        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)

    # Save final model (entire model object for pruning)
    final_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(model, final_path)
    logger.info("Training complete. Best mIoU: %.4f", best_miou)
    logger.info("Final model saved to %s (entire model for pruning)", final_path)
    logger.info("Best state_dict saved to %s", os.path.join(args.output_dir, "best_model.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Segmentation Training")

    # Data
    parser.add_argument("--data-root", type=str, required=True,
                        help="Dataset root with images/ and segs/ subdirectories")
    parser.add_argument("--num-classes", type=int, required=True,
                        help="Number of segmentation classes (including background)")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.1)

    # Model -- modular mode
    parser.add_argument("--backbone", type=str, default=None,
                        choices=list(BACKBONE_REGISTRY.keys()),
                        help="Backbone network (enables modular mode)")
    parser.add_argument("--neck", type=str, default="fpn",
                        choices=NECK_REGISTRY,
                        help="Neck module for feature fusion")
    parser.add_argument("--head", type=str, default="fcn",
                        choices=HEAD_REGISTRY,
                        help="Segmentation head")
    parser.add_argument("--neck-channels", type=int, default=256,
                        help="Output channel dimension of the neck")

    # Model -- preset mode
    parser.add_argument("--model", type=str, default=None,
                        choices=list(PRESET_MODELS.keys()),
                        help="Torchvision preset model (alternative to modular mode)")

    # Model -- common
    parser.add_argument("--pretrained-backbone", action="store_true",
                        help="Initialize backbone with pretrained ImageNet weights")
    parser.add_argument("--aux-loss", action="store_true",
                        help="Use auxiliary loss during training")
    parser.add_argument("--restore", type=str, default=None,
                        help="Path to checkpoint for resuming training")

    # Loss weights (multi-loss: CE + Edge + Lovász)
    parser.add_argument("--ce-weight", type=float, default=1.0,
                        help="Weight for cross-entropy loss (default: 1.0)")
    parser.add_argument("--edge-weight", type=float, default=0.0,
                        help="Weight for edge loss; 0 to disable (default: 0.0)")
    parser.add_argument("--lovasz-weight", type=float, default=0.0,
                        help="Weight for Lovász-Softmax IoU loss; 0 to disable (default: 0.0)")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)

    # Output
    parser.add_argument("--output-dir", type=str, default="output/seg_train")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=10)

    args = parser.parse_args()

    # Validate: at least one mode must be specified
    if args.backbone is None and args.model is None:
        parser.error("Specify either --backbone (modular mode) or --model (preset mode)")

    train(args)
