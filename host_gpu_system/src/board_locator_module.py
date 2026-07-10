"""Warp-space block segmentation locator (auxiliary module for board DQN)."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class BoardLocatorNet(nn.Module):
    """RGB + depth MobileNetV2 features → 112×112 segmentation logits."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.color_features = torchvision.models.mobilenet_v2(
            pretrained=pretrained,
        ).features
        self.depth_features = torchvision.models.mobilenet_v2(
            pretrained=pretrained,
        ).features
        old = self.depth_features[0][0]
        self.depth_features[0][0] = nn.Conv2d(
            1, old.out_channels, kernel_size=old.kernel_size,
            stride=old.stride, padding=old.padding, bias=False,
        )
        if pretrained:
            with torch.no_grad():
                self.depth_features[0][0].weight.copy_(old.weight.mean(dim=1, keepdim=True))

        n_features = 1280 * 2
        self.seg_head = nn.Sequential(OrderedDict([
            ("loc-norm0", nn.BatchNorm2d(n_features)),
            ("loc-relu0", nn.ReLU(inplace=True)),
            ("loc-conv0", nn.Conv2d(n_features, 64, kernel_size=1, stride=1, bias=False)),
            ("loc-norm1", nn.BatchNorm2d(64)),
            ("loc-relu1", nn.ReLU(inplace=True)),
            ("loc-conv1", nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)),
            ("loc-upsam", nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)),
        ]))
        for m in self.seg_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        color_f = self.color_features(rgb)
        depth_f = self.depth_features(depth)
        features = torch.cat((color_f, depth_f), dim=1)
        return self.seg_head(features)


class BoardLocatorModule(nn.Module):
    """Supervised block segmentation on the warped board grid."""

    def __init__(
        self,
        n_cells: int = 12544,
        grid_n: int = 112,
        learning_rate: float = 1e-3,
        weight_decay: float = 8e-5,
        encoder_lr_factor: float = 0.1,
        invalid_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.net = BoardLocatorNet(pretrained=True)
        self.n_cells = n_cells
        self.grid_n = grid_n
        ph = encoder_lr_factor
        self.optimizer = torch.optim.Adam([
            {"params": self.net.color_features.parameters(), "lr": learning_rate * ph,
             "weight_decay": weight_decay * ph},
            {"params": self.net.depth_features.parameters(), "lr": learning_rate * ph,
             "weight_decay": weight_decay * ph},
            {"params": self.net.seg_head.parameters(), "lr": learning_rate,
             "weight_decay": weight_decay},
        ], lr=0.0)
        self.register_buffer(
            "invalid_mask",
            invalid_mask if invalid_mask is not None else torch.zeros(n_cells, dtype=torch.bool),
        )

    def seg_logits(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        return self.net(rgb, depth)

    @torch.no_grad()
    def seg_prob(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.seg_logits(rgb, depth))

    def update(self, batch: Dict) -> Dict[str, float]:
        rgb, depth = batch["rgb"], batch["depth"]
        targets = batch["masks"]
        logits = self.seg_logits(rgb, depth)
        if targets.shape[-2:] != logits.shape[-2:]:
            targets = F.interpolate(
                targets, size=logits.shape[-2:], mode="bilinear", align_corners=True,
            )
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        self.optimizer.step()
        with torch.no_grad():
            pred = (torch.sigmoid(logits) > 0.5).float()
            inter = (pred * targets).sum()
            union = pred.sum() + targets.sum() - inter
            dice = (2.0 * inter / union.clamp(min=1.0)).item()
        return {
            "total": loss.item(),
            "seg_bce": loss.item(),
            "seg_dice": dice,
            "grad_norm": grad_norm.item(),
        }

    @torch.no_grad()
    def predict_mask(self, rgb: torch.Tensor, depth: torch.Tensor) -> np.ndarray:
        prob = self.seg_prob(rgb, depth).squeeze().cpu().numpy()
        return prob.astype(np.float32)

    @torch.no_grad()
    def predict_cell(self, rgb: torch.Tensor, depth: torch.Tensor) -> Tuple[int, float, np.ndarray]:
        mask = self.predict_mask(rgb, depth)
        return cell_from_mask(mask, self.invalid_mask.cpu().numpy(), self.grid_n)

    def save_model(self, filepath: str, training_step: int = 0):
        torch.save({
            "net_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_step": training_step,
            "n_cells": self.n_cells,
            "grid_n": self.grid_n,
        }, filepath)

    def load_model(self, filepath: str) -> int:
        checkpoint = torch.load(filepath, map_location=next(self.parameters()).device)
        self.net.load_state_dict(checkpoint["net_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"   ↳ Board locator optimizer not restored: {e}")
        return int(checkpoint.get("training_step", 0))

    def load_encoder_from_board_dqn(self, board_dqn_module) -> None:
        """Optional warm-start from a FullBoardDQNModule checkpoint."""
        self.net.color_features.load_state_dict(board_dqn_module.net.color_features.state_dict())
        self.net.depth_features.load_state_dict(board_dqn_module.net.depth_features.state_dict())


def cell_from_mask(
    mask: np.ndarray,
    invalid_mask: Optional[np.ndarray],
    grid_n: int = 112,
) -> Tuple[int, float, np.ndarray]:
    """Pool mask into grid cells; return (centroid cell, confidence, mask_grid)."""
    import cv2
    m = np.asarray(mask, dtype=np.float64)
    if m.shape != (grid_n, grid_n):
        m = cv2.resize(m.astype(np.float32), (grid_n, grid_n), interpolation=cv2.INTER_AREA)

    w = np.clip(m, 0.0, None)
    # Focus on the dominant blob: ignore low-probability background.
    peak = float(w.max())
    if peak > 1e-9:
        w = np.where(w >= 0.5 * peak, w, 0.0)
    flat = w.reshape(-1)
    if invalid_mask is not None:
        inv = np.asarray(invalid_mask, dtype=bool).reshape(-1)
        flat = flat.copy()
        flat[inv] = 0.0

    total = float(flat.sum())
    if total <= 1e-9:
        return 0, 0.0, m.astype(np.float32)
    rows = np.repeat(np.arange(grid_n), grid_n)
    cols = np.tile(np.arange(grid_n), grid_n)
    cr = int(round(float((flat * rows).sum() / total)))
    cc = int(round(float((flat * cols).sum() / total)))
    cr = min(max(cr, 0), grid_n - 1)
    cc = min(max(cc, 0), grid_n - 1)
    cell = cr * grid_n + cc
    conf = float(peak)
    return cell, conf, m.astype(np.float32)


def fuse_q_with_detector(
    q_map: np.ndarray,
    detector_cell: int,
    dqn_cell: int,
    invalid_mask: np.ndarray,
    *,
    grid_n: int = 112,
    radius_cells: int = 3,
    min_confidence: float = 0.0,
    detector_confidence: float = 1.0,
) -> Tuple[int, bool]:
    """Snap DQN argmax to best Q within radius of detector_cell."""
    if detector_confidence < min_confidence:
        return int(dqn_cell), False

    q_flat = np.asarray(q_map, dtype=np.float64).reshape(-1)
    inv = np.asarray(invalid_mask, dtype=bool).reshape(-1)
    dr, dc = divmod(int(detector_cell), grid_n)

    best_cell = int(dqn_cell)
    best_q = -math.inf
    found = False
    for cell in range(grid_n * grid_n):
        if inv[cell]:
            continue
        r, c = divmod(cell, grid_n)
        if max(abs(r - dr), abs(c - dc)) > int(radius_cells):
            continue
        qv = float(q_flat[cell])
        if qv > best_q:
            best_q = qv
            best_cell = cell
            found = True

    if not found:
        return int(dqn_cell), False
    return best_cell, True


def create_board_locator_module(
    n_cells: int = 12544,
    grid_n: int = 112,
    learning_rate: float = 1e-3,
    weight_decay: float = 8e-5,
    encoder_lr_factor: float = 0.1,
    invalid_mask: Optional[torch.Tensor] = None,
) -> BoardLocatorModule:
    return BoardLocatorModule(
        n_cells=n_cells,
        grid_n=grid_n,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        encoder_lr_factor=encoder_lr_factor,
        invalid_mask=invalid_mask,
    )
