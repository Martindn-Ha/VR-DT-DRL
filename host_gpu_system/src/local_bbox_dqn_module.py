"""Local-in-window DQN: dual MobileNet → fixed N×N Q-map on a YOLO-centered crop."""

from __future__ import annotations

import copy
import random
from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class LocalBBoxDQNNet(nn.Module):
    """RGB + depth MobileNetV2 features → grid_n×grid_n Q-values."""

    def __init__(self, grid_n: int = 20, pretrained: bool = True):
        super().__init__()
        self.grid_n = int(grid_n)
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
        self.q_head = nn.Sequential(OrderedDict([
            ("lb-norm0", nn.BatchNorm2d(n_features)),
            ("lb-relu0", nn.ReLU(inplace=True)),
            ("lb-conv0", nn.Conv2d(n_features, 64, kernel_size=1, stride=1, bias=False)),
            ("lb-norm1", nn.BatchNorm2d(64)),
            ("lb-relu1", nn.ReLU(inplace=True)),
            ("lb-conv1", nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)),
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
        q = self.q_head(features)
        return F.interpolate(
            q, size=(self.grid_n, self.grid_n), mode="bilinear", align_corners=True,
        )


class LocalBBoxDQNModule(nn.Module):
    """Shaping + DQN on a fixed local window grid."""

    def __init__(
        self,
        grid_n: int = 20,
        learning_rate: float = 1e-3,
        weight_decay: float = 8e-5,
        encoder_lr_factor: float = 0.1,
        gamma: float = 0.99,
    ):
        super().__init__()
        self.grid_n = int(grid_n)
        self.n_cells = self.grid_n * self.grid_n
        self.gamma = gamma
        self.net = LocalBBoxDQNNet(grid_n=self.grid_n, pretrained=True)
        self.net_target = copy.deepcopy(self.net)
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

    def q_map(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        return self.net(rgb, depth)

    def q_flat(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        return self.q_map(rgb, depth).view(rgb.shape[0], -1)

    @torch.no_grad()
    def q_flat_target(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        return self.net_target(rgb, depth).view(rgb.shape[0], -1)

    @torch.no_grad()
    def select_cell(self, rgb: torch.Tensor, depth: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
        batch = rgb.shape[0]
        if epsilon > 0 and random.random() < epsilon:
            return torch.randint(0, self.n_cells, (batch,), device=rgb.device)
        return self.q_flat(rgb, depth).argmax(dim=1)

    def update_shaping(self, batch: Dict) -> Dict[str, float]:
        rgb, depth = batch["rgb"], batch["depth"]
        labels = batch["cell_labels"].long()
        logits = self.q_flat(rgb, depth)
        loss = F.cross_entropy(logits, labels)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        self.optimizer.step()
        with torch.no_grad():
            acc = (logits.argmax(dim=1) == labels).float().mean().item()
        return {
            "total": loss.item(),
            "ce": loss.item(),
            "acc": acc,
            "grad_norm": float(grad_norm),
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
        return {"total": loss.item(), "dqn": loss.item(), "grad_norm": float(grad_norm)}

    def sync_target(self):
        self.net_target.load_state_dict(self.net.state_dict())

    def save_model(self, filepath: str, training_step: int = 0):
        torch.save({
            "net_state_dict": self.net.state_dict(),
            "net_target_state_dict": self.net_target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_step": training_step,
            "dqn_step": self._dqn_step,
            "grid_n": self.grid_n,
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
                print(f"   ↳ Local-bbox DQN optimizer not restored: {e}")
        self._dqn_step = int(checkpoint.get("dqn_step", 0))
        return int(checkpoint.get("training_step", 0))


def create_local_bbox_dqn_module(
    grid_n: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 8e-5,
    encoder_lr_factor: float = 0.1,
    gamma: float = 0.99,
) -> LocalBBoxDQNModule:
    return LocalBBoxDQNModule(
        grid_n=grid_n,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        encoder_lr_factor=encoder_lr_factor,
        gamma=gamma,
    )
