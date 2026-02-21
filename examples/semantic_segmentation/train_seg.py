"""
Semantic Segmentation Training Script

Train a DeepLabV3/FCN segmentation model on a custom dataset.

Dataset structure:
    data_root/
        images/    # original images (.jpg)
        segs/      # segmentation masks (.png), pixel values = class indices

Usage:
    python train_seg.py --data-root /path/to/dataset --num-classes 21 --model deeplabv3_resnet50
    python train_seg.py --data-root /path/to/dataset --num-classes 21 --model fcn_resnet50 --epochs 100
    python train_seg.py --data-root /path/to/dataset --num-classes 21 --model deeplabv3_mobilenet_v3_large
"""

import os
import sys
import argparse
import logging

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    deeplabv3_mobilenet_v3_large,
    fcn_resnet50,
    fcn_resnet101,
    lraspp_mobilenet_v3_large,
)

MODEL_DICT = {
    "deeplabv3_resnet50": deeplabv3_resnet50,
    "deeplabv3_resnet101": deeplabv3_resnet101,
    "deeplabv3_mobilenet_v3_large": deeplabv3_mobilenet_v3_large,
    "fcn_resnet50": fcn_resnet50,
    "fcn_resnet101": fcn_resnet101,
    "lraspp_mobilenet_v3_large": lraspp_mobilenet_v3_large,
}


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

        # Collect matched pairs by filename stem
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

        # Normalization (ImageNet statistics)
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
            # Random resize crop
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
            # Random horizontal flip
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
            continue  # skip classes not present
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


def train(args):
    logger = get_logger("train_seg", log_file=os.path.join(args.output_dir, "train.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Dataset
    full_dataset = SegDataset(args.data_root, image_size=args.image_size, is_train=True)
    val_size = max(1, int(len(full_dataset) * args.val_ratio))
    train_size = len(full_dataset) - val_size
    train_dst, val_dst = random_split(full_dataset, [train_size, val_size])
    # Override is_train for val split items (use eval transforms via separate dataset)
    val_eval_dst = SegDataset(args.data_root, image_size=args.image_size, is_train=False)
    # Use same indices as val_dst
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
    if args.model not in MODEL_DICT:
        raise ValueError(f"Unknown model: {args.model}. Choose from {list(MODEL_DICT.keys())}")

    if args.model.startswith("lraspp"):
        model = MODEL_DICT[args.model](num_classes=args.num_classes)
    else:
        model = MODEL_DICT[args.model](num_classes=args.num_classes, aux_loss=args.aux_loss)

    if args.pretrained_backbone:
        logger.info("Using pretrained backbone weights")
        # Load a pretrained model and transfer backbone weights
        if args.model.startswith("lraspp"):
            pretrained = MODEL_DICT[args.model](num_classes=21)
        else:
            pretrained = MODEL_DICT[args.model](num_classes=21, aux_loss=True)
        # Copy backbone
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
            loss = F.cross_entropy(output["out"], masks, ignore_index=255)
            if "aux" in output and output["aux"] is not None and args.aux_loss:
                loss += 0.4 * F.cross_entropy(output["aux"], masks, ignore_index=255)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % args.log_interval == 0:
                logger.info(
                    "Epoch [%d/%d] Iter [%d/%d] Loss: %.4f LR: %.6f",
                    epoch + 1, args.epochs, i + 1, len(train_loader),
                    running_loss / (i + 1), optimizer.param_groups[0]["lr"],
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

        # Periodic checkpoint
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
    parser.add_argument("--image-size", type=int, default=512, help="Training image size")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")

    # Model
    parser.add_argument("--model", type=str, default="deeplabv3_resnet50",
                        choices=list(MODEL_DICT.keys()),
                        help="Segmentation model architecture")
    parser.add_argument("--pretrained-backbone", action="store_true",
                        help="Initialize backbone with pretrained ImageNet weights")
    parser.add_argument("--aux-loss", action="store_true",
                        help="Use auxiliary loss during training")
    parser.add_argument("--restore", type=str, default=None,
                        help="Path to checkpoint for resuming training")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)

    # Output
    parser.add_argument("--output-dir", type=str, default="output/seg_train",
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=10)

    args = parser.parse_args()
    train(args)
