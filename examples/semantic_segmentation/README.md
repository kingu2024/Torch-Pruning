# Semantic Segmentation: Training & Pruning

This example demonstrates how to train and prune semantic segmentation models (DeepLabV3, FCN, LRASPP) using **Torch-Pruning**.

## Dataset Structure

```
data_root/
├── images/          # Original images (.jpg)
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
└── segs/            # Segmentation masks (.png)
    ├── 001.png      # Pixel values = class indices (0, 1, 2, ...)
    ├── 002.png
    └── ...
```

- Image and mask filenames must share the same stem (e.g., `001.jpg` and `001.png`).
- Mask pixel values represent class indices. Use `255` for ignore/unlabeled regions.

## Supported Models

| Model | Key |
|---|---|
| DeepLabV3-ResNet50 | `deeplabv3_resnet50` |
| DeepLabV3-ResNet101 | `deeplabv3_resnet101` |
| DeepLabV3-MobileNetV3 | `deeplabv3_mobilenet_v3_large` |
| FCN-ResNet50 | `fcn_resnet50` |
| FCN-ResNet101 | `fcn_resnet101` |
| LRASPP-MobileNetV3 | `lraspp_mobilenet_v3_large` |

## Supported Pruning Methods

| Method | Key | Sparsity Learning |
|---|---|---|
| L1 Magnitude | `l1` | No |
| L2 Magnitude | `l2` | No |
| FPGM | `fpgm` | No |
| LAMP | `lamp` | No |
| Random | `random` | No |
| Taylor | `taylor` | No |
| BN Slim | `slim` | Yes |
| Group BN Slim | `group_slim` | Yes |
| Group Norm (DepGraph) | `group_norm` | No |
| Group SL | `group_sl` | Yes |
| Growing Reg | `growing_reg` | Yes |

## Step 1: Training

```bash
python train_seg.py \
    --data-root /path/to/dataset \
    --num-classes 21 \
    --model deeplabv3_resnet50 \
    --pretrained-backbone \
    --epochs 100 \
    --batch-size 8 \
    --lr 0.01 \
    --output-dir output/seg_train
```

## Step 2: Pruning

### One-shot pruning (no sparsity learning)

```bash
python prune_seg.py \
    --data-root /path/to/dataset \
    --num-classes 21 \
    --model deeplabv3_resnet50 \
    --restore output/seg_train/final_model.pth \
    --method l1 \
    --speed-up 2.0 \
    --global-pruning \
    --finetune \
    --finetune-epochs 50 \
    --output-dir output/seg_prune
```

### Pruning with sparsity learning (e.g., group_norm)

```bash
python prune_seg.py \
    --data-root /path/to/dataset \
    --num-classes 21 \
    --model deeplabv3_resnet50 \
    --restore output/seg_train/final_model.pth \
    --method group_norm \
    --speed-up 2.0 \
    --sl-epochs 50 \
    --finetune \
    --finetune-epochs 50 \
    --output-dir output/seg_prune
```

## Output Files

After training:
- `output/seg_train/best_model.pth` — Best state_dict checkpoint
- `output/seg_train/final_model.pth` — Final model (entire `nn.Module`)

After pruning:
- `output/seg_prune/pruned_model.pth` — Pruned model (before finetuning)
- `output/seg_prune/pruned_finetuned_best.pth` — Best finetuned pruned model
