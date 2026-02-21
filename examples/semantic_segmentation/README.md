# Semantic Segmentation: Training & Pruning

This example demonstrates how to train and prune semantic segmentation models using **Torch-Pruning**.

Two modes are supported:
- **Modular mode**: freely compose your model from `backbone + neck + head`
- **Preset mode**: use torchvision segmentation models directly (DeepLabV3, FCN, LRASPP)

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

## Modular Architecture

Build your segmentation model by combining any backbone, neck, and head:

| Component | Options | Description |
|-----------|---------|-------------|
| **Backbone** | `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152` | ResNet family |
| | `mobilenet_v2`, `mobilenet_v3_large`, `mobilenet_v3_small` | MobileNet family |
| **Neck** | `fpn` | Feature Pyramid Network (multi-scale fusion) |
| | `aspp` | Atrous Spatial Pyramid Pooling (DeepLabV3 style) |
| | `ppm` | Pyramid Pooling Module (PSPNet style) |
| | `identity` | Pass-through (no fusion, uses deepest feature) |
| **Head** | `fcn` | Conv-BN-ReLU-Dropout-Conv |
| | `simple` | Single 1x1 convolution |

## Preset Models

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

### Modular mode (recommended)

```bash
python train_seg.py \
    --data-root /path/to/dataset \
    --num-classes 21 \
    --backbone resnet50 \
    --neck fpn \
    --head fcn \
    --pretrained-backbone \
    --epochs 100 \
    --batch-size 8 \
    --output-dir output/seg_train
```

Other combinations:

```bash
# DeepLab-style: ResNet101 + ASPP + FCN head
python train_seg.py --data-root /data --num-classes 21 \
    --backbone resnet101 --neck aspp --head fcn --pretrained-backbone

# PSPNet-style: ResNet50 + PPM + FCN head
python train_seg.py --data-root /data --num-classes 21 \
    --backbone resnet50 --neck ppm --head fcn --pretrained-backbone

# Lightweight: MobileNetV3 + FPN + simple head
python train_seg.py --data-root /data --num-classes 21 \
    --backbone mobilenet_v3_large --neck fpn --head simple --pretrained-backbone
```

### Preset mode (torchvision models)

```bash
python train_seg.py \
    --data-root /path/to/dataset \
    --num-classes 21 \
    --model deeplabv3_resnet50 \
    --pretrained-backbone \
    --epochs 100 \
    --output-dir output/seg_train
```

## Step 2: Pruning

### Modular model

```bash
python prune_seg.py \
    --data-root /path/to/dataset \
    --num-classes 21 \
    --backbone resnet50 --neck fpn --head fcn \
    --restore output/seg_train/final_model.pth \
    --method l1 \
    --speed-up 2.0 \
    --global-pruning \
    --finetune \
    --finetune-epochs 50 \
    --output-dir output/seg_prune
```

### Preset model

```bash
python prune_seg.py \
    --data-root /path/to/dataset \
    --num-classes 21 \
    --model deeplabv3_resnet50 \
    --restore output/seg_train/final_model.pth \
    --method group_norm \
    --speed-up 2.0 \
    --finetune \
    --finetune-epochs 50 \
    --output-dir output/seg_prune
```

### Pruning with sparsity learning (e.g., slim)

```bash
python prune_seg.py \
    --data-root /path/to/dataset \
    --num-classes 21 \
    --backbone resnet50 --neck aspp --head fcn \
    --restore output/seg_train/final_model.pth \
    --method slim \
    --speed-up 2.0 \
    --sl-epochs 50 \
    --finetune \
    --finetune-epochs 50
```

## Output Files

After training:
- `output/seg_train/best_model.pth` — Best state_dict checkpoint
- `output/seg_train/final_model.pth` — Final model (entire `nn.Module`)

After pruning:
- `output/seg_prune/pruned_model.pth` — Pruned model (before finetuning)
- `output/seg_prune/pruned_finetuned_best.pth` — Best finetuned pruned model
