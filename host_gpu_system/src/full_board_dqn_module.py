"""Full-board spatial DQN (Gomes et al.): dual MobileNet encoders → 112×112 Q-map."""

from __future__ import annotations

import copy
import random
from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class FullBoardDQNNet(nn.Module):
    """RGB + depth MobileNetV2 features → 112×112 Q-values."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.color_features = torchvision.models.mobilenet_v2(
            pretrained=pretrained,
        ).features
        self.depth_features = torchvision.models.mobilenet_v2(
            pretrained=pretrained,
        ).features
        # Depth 1ch → 3ch for ImageNet stem
        old = self.depth_features[0][0]
        self.depth_features[0][0] = nn.Conv2d(
            1, old.out_channels, kernel_size=old.kernel_size,
            stride=old.stride, padding=old.padding, bias=False,
        )
        if pretrained:
            with torch.no_grad():
                self.depth_features[0][0].weight.copy_(old.weight.mean(dim=1, keepdim=True))

        n_features = 1280 * 2
        self.q_head = nn.Sequential(OrderedDict([
            ("board-norm0", nn.BatchNorm2d(n_features)),
            ("board-relu0", nn.ReLU(inplace=True)),
            ("board-conv0", nn.Conv2d(n_features, 64, kernel_size=1, stride=1, bias=False)),
            ("board-norm1", nn.BatchNorm2d(64)),
            ("board-relu1", nn.ReLU(inplace=True)),
            ("board-conv1", nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)),
            ("board-upsam", nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)),
        ]))
        for m in self.q_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        color_f = self.color_features(rgb)
        depth_f = self.depth_features(depth)
        features = torch.cat((color_f, depth_f), dim=1)
        return self.q_head(features)


class FullBoardDQNModule(nn.Module):
    """Shaping + DQN on 112×112 full-board grid."""

    def __init__(
        self,
        n_cells: int = 12544,
        learning_rate: float = 1e-3,
        weight_decay: float = 8e-5,
        encoder_lr_factor: float = 0.1,
        gamma: float = 0.99,
        invalid_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.net = FullBoardDQNNet(pretrained=True)
        self.net_target = copy.deepcopy(self.net)
        self.n_cells = n_cells
        self.gamma = gamma
        ph = encoder_lr_factor
        self.optimizer = torch.optim.Adam([
            {"params": self.net.color_features.parameters(), "lr": learning_rate * ph,
             "weight_decay": weight_decay * ph},
            {"params": self.net.depth_features.parameters(), "lr": learning_rate * ph,
             "weight_decay": weight_decay * ph},
            {"params": self.net.q_head.parameters(), "lr": learning_rate,
             "weight_decay": weight_decay},
        ], lr=0.0)
        self._dqn_step = 0
        self.register_buffer(
            "invalid_mask",
            invalid_mask if invalid_mask is not None else torch.zeros(n_cells, dtype=torch.bool),
        )

    def q_map(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        return self.net(rgb, depth)

    def q_flat(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        q = self.q_map(rgb, depth).view(rgb.shape[0], -1)
        if self.invalid_mask.any():
            q = q.masked_fill(self.invalid_mask.unsqueeze(0), -1e9)
        return q

    @torch.no_grad()
    def q_flat_target(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        q = self.net_target(rgb, depth).view(rgb.shape[0], -1)
        if self.invalid_mask.any():
            q = q.masked_fill(self.invalid_mask.unsqueeze(0), -1e9)
        return q

    @torch.no_grad()
    def select_cell(self, rgb: torch.Tensor, depth: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
        batch = rgb.shape[0]
        if epsilon > 0 and random.random() < epsilon:
            valid = (~self.invalid_mask).nonzero(as_tuple=False).squeeze(-1)
            if valid.numel() == 0:
                return torch.randint(0, self.n_cells, (batch,), device=rgb.device)
            picks = valid[torch.randint(0, valid.numel(), (batch,), device=rgb.device)]
            return picks
        return self.q_flat(rgb, depth).argmax(dim=1)

    def update_shaping(self, batch: Dict) -> Dict[str, float]:
        rgb, depth = batch["rgb"], batch["depth"]
        targets = batch["q_targets"]
        q = self.q_map(rgb, depth).view(rgb.shape[0], -1)
        valid = (~self.invalid_mask).float().unsqueeze(0)
        diff = F.smooth_l1_loss(q, targets, reduction="none")
        masked = diff * valid
        denom = valid.expand_as(diff).sum().clamp(min=1.0)
        loss = masked.sum() / denom
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        self.optimizer.step()
        with torch.no_grad():
            mse = ((q - targets) ** 2 * valid).sum() / denom
        return {
            "total": loss.item(),
            "shaping_mse": mse.item(),
            "grad_norm": grad_norm.item(),
        }

    def update_dqn(self, batch: Dict) -> Dict[str, float]:
        rgb, depth = batch["rgb"], batch["depth"]
        next_rgb, next_depth = batch["next_rgb"], batch["next_depth"]
        actions = batch["cell_actions"].long()
        rewards = batch["rewards"].view(-1, 1)
        dones = batch["dones"].view(-1, 1)

        q = self.q_flat(rgb, depth).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q = self.q_flat_target(next_rgb, next_depth).max(dim=1, keepdim=True)[0]
            target = rewards + (1.0 - dones) * self.gamma * next_q
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        self.optimizer.step()
        self._dqn_step += 1
        return {"total": loss.item(), "dqn": loss.item(), "grad_norm": grad_norm.item()}

    def sync_target(self):
        self.net_target.load_state_dict(self.net.state_dict())

    def save_model(self, filepath: str, training_step: int = 0):
        torch.save({
            "net_state_dict": self.net.state_dict(),
            "net_target_state_dict": self.net_target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_step": training_step,
            "dqn_step": self._dqn_step,
            "n_cells": self.n_cells,
        }, filepath)

    def load_model(self, filepath: str) -> int:
        checkpoint = torch.load(filepath, map_location=next(self.parameters()).device)
        self.net.load_state_dict(checkpoint["net_state_dict"])
        self.net_target.load_state_dict(checkpoint.get(
            "net_target_state_dict", checkpoint["net_state_dict"],
        ))
        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"   ↳ Board DQN optimizer not restored: {e}")
        self._dqn_step = int(checkpoint.get("dqn_step", 0))
        return int(checkpoint.get("training_step", 0))


def create_full_board_dqn_module(
    n_cells: int = 12544,
    learning_rate: float = 1e-3,
    weight_decay: float = 8e-5,
    encoder_lr_factor: float = 0.1,
    gamma: float = 0.99,
    invalid_mask: Optional[torch.Tensor] = None,
) -> FullBoardDQNModule:
    return FullBoardDQNModule(
        n_cells=n_cells,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        encoder_lr_factor=encoder_lr_factor,
        gamma=gamma,
        invalid_mask=invalid_mask,
    )
