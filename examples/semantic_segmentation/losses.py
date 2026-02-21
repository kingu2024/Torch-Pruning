"""
Multi-Loss Functions for Semantic Segmentation

Provides:
    - EdgeLoss:       Sobel-based edge extraction loss for boundary refinement
    - LovaszSoftmax:  Lovász extension of IoU loss for direct IoU optimization
    - MultiSegLoss:   Composite loss combining CE + Edge + Lovász with configurable weights

Usage:
    criterion = MultiSegLoss(
        num_classes=21,
        ce_weight=1.0,
        edge_weight=0.5,
        lovasz_weight=0.5,
        ignore_index=255,
    )
    loss = criterion(logits, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# Edge Extraction Loss
# =====================================================

class EdgeLoss(nn.Module):
    """Edge-aware loss using Sobel filters.

    Extracts edges from both ground-truth masks and predicted segmentation,
    then computes binary cross-entropy between the two edge maps.  This
    encourages the model to produce sharper boundaries.
    """

    def __init__(self, num_classes, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # Sobel kernels (fixed, not learnable)
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _extract_edges(self, mask):
        """Extract binary edge map from a single-channel mask using Sobel filters.

        Args:
            mask: (B, 1, H, W) float tensor
        Returns:
            edge: (B, 1, H, W) binary edge map
        """
        gx = F.conv2d(mask, self.sobel_x, padding=1)
        gy = F.conv2d(mask, self.sobel_y, padding=1)
        edge = (gx.abs() + gy.abs()).clamp(0, 1)
        return (edge > 0).float()

    def forward(self, logits, targets):
        """
        Args:
            logits:  (B, C, H, W) raw model output
            targets: (B, H, W) long tensor with class indices
        """
        device = logits.device
        B, C, H, W = logits.shape

        # Build valid-pixel mask (True where target is valid)
        valid = (targets != self.ignore_index)  # (B, H, W)

        # Compute predicted hard segmentation
        pred = logits.argmax(dim=1)  # (B, H, W)

        # Edge extraction on ground-truth
        gt_float = targets.clone().float()
        gt_float[~valid] = 0  # zero out ignored pixels
        gt_edges = self._extract_edges(gt_float.unsqueeze(1))  # (B, 1, H, W)

        # Edge extraction on prediction
        pred_float = pred.float()
        pred_edges = self._extract_edges(pred_float.unsqueeze(1))  # (B, 1, H, W)

        # Soft prediction edges: use softmax probabilities for gradient flow.
        # We compute a soft "gradient magnitude" from per-class probability maps
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        # Compute Sobel magnitude per class, then take max across classes
        sobel_x_exp = self.sobel_x.expand(C, -1, -1, -1)  # (C, 1, 3, 3)
        sobel_y_exp = self.sobel_y.expand(C, -1, -1, -1)  # (C, 1, 3, 3)
        gx = F.conv2d(probs, sobel_x_exp, padding=1, groups=C)  # (B, C, H, W)
        gy = F.conv2d(probs, sobel_y_exp, padding=1, groups=C)  # (B, C, H, W)
        soft_edge = (gx.abs() + gy.abs()).max(dim=1, keepdim=True)[0]  # (B, 1, H, W)
        # Normalize to [0, 1]
        soft_edge = soft_edge / (soft_edge.max() + 1e-8)

        # Apply valid-pixel mask
        valid_mask = valid.unsqueeze(1).float()  # (B, 1, H, W)
        gt_edges = gt_edges * valid_mask
        soft_edge = soft_edge * valid_mask

        # Binary cross-entropy between soft predicted edges and GT edges
        loss = F.binary_cross_entropy(
            soft_edge.clamp(1e-6, 1 - 1e-6), gt_edges, weight=valid_mask, reduction="sum"
        )
        num_valid = valid_mask.sum().clamp(min=1)
        return loss / num_valid


# =====================================================
# Lovász-Softmax Loss
# =====================================================

def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovász extension w.r.t sorted errors.

    Reference: "The Lovász-Softmax loss" (Berman et al., CVPR 2018).
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def _flatten_probas(probas, labels, ignore_index=None):
    """Flatten predictions and labels, removing ignored pixels.

    Args:
        probas: (B, C, H, W) class probabilities
        labels: (B, H, W) integer class labels
        ignore_index: label value to ignore

    Returns:
        vprobas: (P, C) valid probabilities
        vlabels: (P,) valid labels
    """
    if probas.dim() == 3:
        probas = probas.unsqueeze(1)
    B, C, H, W = probas.shape
    probas = probas.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
    labels = labels.view(-1)  # (B*H*W,)
    if ignore_index is not None:
        valid = labels != ignore_index
        probas = probas[valid]
        labels = labels[valid]
    return probas, labels


def _lovasz_softmax_flat(probas, labels, classes="present"):
    """Multi-class Lovász-Softmax loss on flattened predictions.

    Args:
        probas: (P, C) class probabilities
        labels: (P,) ground truth class indices
        classes: 'all' for all, 'present' for classes present in labels
    """
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.shape[1]
    losses = []
    for c in range(C):
        fg = (labels == c).float()  # foreground for class c
        if classes == "present" and fg.sum() == 0:
            continue
        if C == 1:
            fg_class = 1.0 - probas[:, 0]
        else:
            fg_class = probas[:, c]
        errors = (fg - fg_class).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.detach()
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    if len(losses) == 0:
        return torch.tensor(0.0, device=probas.device, requires_grad=True)
    return torch.stack(losses).mean()


class LovaszSoftmax(nn.Module):
    """Lovász-Softmax loss for semantic segmentation.

    Directly optimizes the mean IoU metric through a convex surrogate.
    Reference: Berman et al., "The Lovász-Softmax loss: A tractable surrogate
    for the optimization of the intersection-over-union measure in neural
    networks", CVPR 2018.
    """

    def __init__(self, classes="present", ignore_index=255):
        super().__init__()
        self.classes = classes
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits:  (B, C, H, W) raw model output
            targets: (B, H, W) long tensor with class indices
        """
        probas = F.softmax(logits, dim=1)
        vprobas, vlabels = _flatten_probas(probas, targets, self.ignore_index)
        return _lovasz_softmax_flat(vprobas, vlabels, classes=self.classes)


# =====================================================
# Composite Multi-Loss
# =====================================================

class MultiSegLoss(nn.Module):
    """Composite loss for semantic segmentation.

    Combines:
        - Cross-entropy loss (pixel classification)
        - Edge loss (boundary sharpness)
        - Lovász-Softmax loss (IoU optimization)

    Total = ce_weight * CE + edge_weight * Edge + lovasz_weight * Lovász

    All sub-losses respect ignore_index for void / unlabeled pixels.
    """

    def __init__(self, num_classes, ce_weight=1.0, edge_weight=0.5,
                 lovasz_weight=0.5, ignore_index=255):
        super().__init__()
        self.ce_weight = ce_weight
        self.edge_weight = edge_weight
        self.lovasz_weight = lovasz_weight
        self.ignore_index = ignore_index

        self.edge_loss = EdgeLoss(num_classes, ignore_index) if edge_weight > 0 else None
        self.lovasz_loss = LovaszSoftmax(ignore_index=ignore_index) if lovasz_weight > 0 else None

    def forward(self, logits, targets):
        """
        Args:
            logits:  (B, C, H, W) raw model output
            targets: (B, H, W) long tensor with class indices

        Returns:
            total_loss: weighted sum of losses
            loss_dict:  dict with individual loss values for logging
        """
        loss_dict = {}

        # 1. Cross-entropy loss
        ce = F.cross_entropy(logits, targets, ignore_index=self.ignore_index)
        loss_dict["ce"] = ce.item()
        total = self.ce_weight * ce

        # 2. Edge loss
        if self.edge_loss is not None:
            edge = self.edge_loss(logits, targets)
            loss_dict["edge"] = edge.item()
            total = total + self.edge_weight * edge

        # 3. Lovász-Softmax loss
        if self.lovasz_loss is not None:
            lovasz = self.lovasz_loss(logits, targets)
            loss_dict["lovasz"] = lovasz.item()
            total = total + self.lovasz_weight * lovasz

        loss_dict["total"] = total.item()
        return total, loss_dict
