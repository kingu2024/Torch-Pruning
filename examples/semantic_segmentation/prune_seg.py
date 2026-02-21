"""
Semantic Segmentation Pruning Script

Prune a trained segmentation model using Torch-Pruning, with optional
sparsity-learning regularization and post-pruning finetuning.

Supports both modular (backbone+neck+head) and preset torchvision models.

Pruning methods:
    - l1 / l2           : Magnitude-based pruning
    - fpgm              : Filter Pruning via Geometric Median
    - lamp              : Layer-Adaptive Magnitude Pruning
    - slim              : BN Scaling Factor (requires sparsity learning)
    - group_slim        : Group-level BN Scaling Factor
    - group_norm        : Group Norm regularization (DepGraph paper)
    - growing_reg       : Growing Regularization
    - random            : Random pruning (baseline)
    - taylor            : First-order Taylor expansion

Usage:
    # Modular model
    python prune_seg.py --data-root /data --num-classes 21 \
        --backbone resnet50 --neck fpn --head fcn \
        --restore output/seg_train/final_model.pth \
        --method l1 --speed-up 2.0 --finetune --finetune-epochs 50

    # Preset model
    python prune_seg.py --data-root /data --num-classes 21 \
        --model deeplabv3_resnet50 \
        --restore output/seg_train/final_model.pth \
        --method group_norm --speed-up 2.0 --finetune --finetune-epochs 50
"""

import os
import sys

# Allow importing torch_pruning from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import argparse
import logging
from functools import partial

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch_pruning as tp

from train_seg import (
    SegDataset, evaluate, compute_miou, build_model,
    PRESET_MODELS, MODEL_DICT, BACKBONE_REGISTRY, NECK_REGISTRY, HEAD_REGISTRY,
)
from losses import MultiSegLoss


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


def get_pruner(model, example_inputs, args):
    """Build a pruner based on the specified method."""

    sparsity_learning = False

    if args.method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "l1":
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "l2":
        imp = tp.importance.MagnitudeImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "fpgm":
        imp = tp.importance.FPGMImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "lamp":
        imp = tp.importance.LAMPImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "slim":
        sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning)
    elif args.method == "group_slim":
        sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning, group_lasso=True)
    elif args.method == "group_norm":
        imp = tp.importance.GroupMagnitudeImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=args.global_pruning)
    elif args.method == "group_sl":
        sparsity_learning = True
        imp = tp.importance.GroupMagnitudeImportance(p=2, normalizer="max")
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=args.reg, global_pruning=args.global_pruning)
    elif args.method == "growing_reg":
        sparsity_learning = True
        imp = tp.importance.GroupMagnitudeImportance(p=2)
        pruner_entry = partial(tp.pruner.GrowingRegPruner, reg=args.reg, delta_reg=args.delta_reg, global_pruning=args.global_pruning)
    elif args.method == "taylor":
        imp = tp.importance.TaylorImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    else:
        raise ValueError(f"Unknown pruning method: {args.method}")

    # Find layers to ignore (output heads that match num_classes)
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.out_channels == args.num_classes:
            ignored_layers.append(m)

    pruner = pruner_entry(
        model,
        example_inputs=example_inputs,
        importance=imp,
        iterative_steps=args.iterative_steps,
        pruning_ratio=1.0,
        max_pruning_ratio=args.max_pruning_ratio,
        ignored_layers=ignored_layers,
        output_transform=lambda x: x["out"],
    )
    return pruner, sparsity_learning


def progressive_pruning(pruner, model, speed_up, example_inputs):
    """Progressively prune until the target speed-up is reached."""
    model.eval()
    base_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    current_speed_up = 1.0
    while current_speed_up < speed_up:
        pruner.step()
        pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        current_speed_up = float(base_ops) / pruned_ops
        if pruner.current_step == pruner.iterative_steps:
            break
    return current_speed_up


def build_criterion(args, device):
    """Build the multi-loss criterion from CLI arguments."""
    return MultiSegLoss(
        num_classes=args.num_classes,
        ce_weight=args.ce_weight,
        edge_weight=args.edge_weight,
        lovasz_weight=args.lovasz_weight,
        ignore_index=255,
    ).to(device)


def sparsity_learning(model, pruner, train_loader, args, logger, device):
    """Run sparsity learning (regularization) phase before pruning."""
    criterion = build_criterion(args, device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.sl_lr, momentum=0.9,
        weight_decay=0,
    )
    milestones = [int(ms) for ms in args.sl_lr_decay_milestones.split(",")]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    model.to(device)
    for epoch in range(args.sl_epochs):
        model.train()
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss, loss_dict = criterion(output["out"], masks)
            loss.backward()
            pruner.regularize(model)
            optimizer.step()

            if (i + 1) % args.log_interval == 0:
                loss_info = " ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
                logger.info(
                    "  [Sparsity Learning] Epoch [%d/%d] Iter [%d/%d] Loss: %.4f (%s)",
                    epoch + 1, args.sl_epochs, i + 1, len(train_loader),
                    loss.item(), loss_info,
                )

        if isinstance(pruner, tp.pruner.GrowingRegPruner):
            pruner.update_reg()

        scheduler.step()

    logger.info("Sparsity learning complete.")


def finetune(model, train_loader, val_loader, args, logger, device):
    """Finetune the pruned model to recover accuracy."""
    criterion = build_criterion(args, device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.finetune_lr, momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)

    model.to(device)
    best_miou = 0.0
    for epoch in range(args.finetune_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss, loss_dict = criterion(output["out"], masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % args.log_interval == 0:
                loss_info = " ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
                logger.info(
                    "  [Finetune] Epoch [%d/%d] Iter [%d/%d] Loss: %.4f (%s) LR: %.6f",
                    epoch + 1, args.finetune_epochs, i + 1, len(train_loader),
                    running_loss / (i + 1), loss_info, optimizer.param_groups[0]["lr"],
                )
        scheduler.step()

        miou, val_loss = evaluate(model, val_loader, args.num_classes, device)
        logger.info(
            "  [Finetune] Epoch [%d/%d] Val mIoU: %.4f Val Loss: %.4f",
            epoch + 1, args.finetune_epochs, miou, val_loss,
        )
        if miou > best_miou:
            best_miou = miou
            save_path = os.path.join(args.output_dir, "pruned_finetuned_best.pth")
            torch.save(model, save_path)
            logger.info("  Saved best finetuned model (mIoU=%.4f)", best_miou)

    logger.info("Finetuning complete. Best mIoU: %.4f", best_miou)


def main():
    parser = argparse.ArgumentParser(description="Semantic Segmentation Pruning with Torch-Pruning")

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
                        choices=NECK_REGISTRY)
    parser.add_argument("--head", type=str, default="fcn",
                        choices=HEAD_REGISTRY)
    parser.add_argument("--neck-channels", type=int, default=256)

    # Model -- preset mode
    parser.add_argument("--model", type=str, default=None,
                        choices=list(PRESET_MODELS.keys()))

    # Restore
    parser.add_argument("--restore", type=str, required=True,
                        help="Path to trained model checkpoint (entire model .pth or state_dict)")

    # Pruning
    parser.add_argument("--method", type=str, default="l1",
                        choices=["random", "l1", "l2", "fpgm", "lamp", "slim",
                                 "group_slim", "group_norm", "group_sl",
                                 "growing_reg", "taylor"])
    parser.add_argument("--speed-up", type=float, default=2.0,
                        help="Target FLOPs speed-up ratio")
    parser.add_argument("--max-pruning-ratio", type=float, default=1.0)
    parser.add_argument("--iterative-steps", type=int, default=400)
    parser.add_argument("--global-pruning", action="store_true")

    # Regularization (for sparsity-learning methods)
    parser.add_argument("--reg", type=float, default=5e-4)
    parser.add_argument("--delta-reg", type=float, default=1e-4)
    parser.add_argument("--sl-epochs", type=int, default=100)
    parser.add_argument("--sl-lr", type=float, default=0.005)
    parser.add_argument("--sl-lr-decay-milestones", type=str, default="60,80")

    # Loss weights (multi-loss: CE + Edge + Lovász)
    parser.add_argument("--ce-weight", type=float, default=1.0,
                        help="Weight for cross-entropy loss (default: 1.0)")
    parser.add_argument("--edge-weight", type=float, default=0.0,
                        help="Weight for edge loss; 0 to disable (default: 0.0)")
    parser.add_argument("--lovasz-weight", type=float, default=0.0,
                        help="Weight for Lovász-Softmax IoU loss; 0 to disable (default: 0.0)")

    # Finetuning
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--finetune-lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # General
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="output/seg_prune")
    parser.add_argument("--log-interval", type=int, default=10)

    args = parser.parse_args()

    if args.backbone is None and args.model is None:
        parser.error("Specify either --backbone (modular mode) or --model (preset mode)")

    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger("prune_seg", log_file=os.path.join(args.output_dir, "prune.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # =====================================================
    # 1. Load dataset
    # =====================================================
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

    # =====================================================
    # 2. Load trained model
    # =====================================================
    logger.info("Loading model from %s", args.restore)
    loaded = torch.load(args.restore, map_location="cpu")
    if isinstance(loaded, nn.Module):
        model = loaded
    else:
        # state_dict: rebuild the model architecture first
        model = build_model(args)
        model.load_state_dict(loaded)
    model = model.to(device)
    model.eval()

    # =====================================================
    # 3. Evaluate before pruning
    # =====================================================
    example_inputs = torch.randn(1, 3, args.image_size, args.image_size).to(device)
    ori_ops, ori_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    ori_miou, ori_loss = evaluate(model, val_loader, args.num_classes, device)
    logger.info("=" * 60)
    logger.info("Before Pruning:")
    logger.info("  Params : %.2f M", ori_params / 1e6)
    logger.info("  FLOPs  : %.2f M", ori_ops / 1e6)
    logger.info("  mIoU   : %.4f", ori_miou)
    logger.info("  Val Loss: %.4f", ori_loss)
    logger.info("=" * 60)

    # =====================================================
    # 4. Build pruner
    # =====================================================
    pruner, need_sparsity_learning = get_pruner(model, example_inputs, args)

    # =====================================================
    # 5. Sparsity learning (if needed)
    # =====================================================
    if need_sparsity_learning and args.sl_epochs > 0:
        logger.info("Starting sparsity learning phase (%s)...", args.method)
        sparsity_learning(model, pruner, train_loader, args, logger, device)

    # =====================================================
    # 6. Pruning
    # =====================================================
    logger.info("Pruning with method=%s, target speed-up=%.1fx ...", args.method, args.speed_up)
    model.eval()
    actual_speedup = progressive_pruning(pruner, model, args.speed_up, example_inputs)
    del pruner

    # =====================================================
    # 7. Evaluate after pruning
    # =====================================================
    pruned_ops, pruned_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    pruned_miou, pruned_loss = evaluate(model, val_loader, args.num_classes, device)
    logger.info("=" * 60)
    logger.info("After Pruning:")
    logger.info("  Params : %.2f M => %.2f M (%.1f%%)",
                ori_params / 1e6, pruned_params / 1e6, pruned_params / ori_params * 100)
    logger.info("  FLOPs  : %.2f M => %.2f M (%.1f%%, %.2fX)",
                ori_ops / 1e6, pruned_ops / 1e6, pruned_ops / ori_ops * 100, ori_ops / pruned_ops)
    logger.info("  mIoU   : %.4f => %.4f", ori_miou, pruned_miou)
    logger.info("  Val Loss: %.4f => %.4f", ori_loss, pruned_loss)
    logger.info("=" * 60)

    # Save pruned model (before finetuning)
    pruned_path = os.path.join(args.output_dir, "pruned_model.pth")
    torch.save(model, pruned_path)
    logger.info("Pruned model saved to %s", pruned_path)

    # =====================================================
    # 8. Finetuning (optional)
    # =====================================================
    if args.finetune:
        logger.info("Starting finetuning for %d epochs...", args.finetune_epochs)
        finetune(model, train_loader, val_loader, args, logger, device)

        final_miou, final_loss = evaluate(model, val_loader, args.num_classes, device)
        logger.info("=" * 60)
        logger.info("After Finetuning:")
        logger.info("  mIoU   : %.4f => %.4f => %.4f (orig => pruned => finetuned)",
                     ori_miou, pruned_miou, final_miou)
        logger.info("=" * 60)

    logger.info("Done.")


if __name__ == "__main__":
    main()
