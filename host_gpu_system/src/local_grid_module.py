"""Local grid Q-network module (separate file to keep enhanced_neural_network lean)."""

import copy
import random
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from enhanced_neural_network import UR3GraspCNN_Enhanced


class LocalGridQHead(nn.Module):
    """Q-values per local grid cell from pooled visual features."""

    def __init__(self, feature_size: int = 1280, hidden: int = 256, n_cells: int = 25):
        super().__init__()
        self.n_cells = n_cells
        self.net = nn.Sequential(
            nn.Linear(feature_size, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_cells),
        )
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class LocalGridModule(nn.Module):
    """Local grid cell picker: CNN features -> Q per cell (shaping + DQN)."""

    def __init__(self,
                 grasp_net: UR3GraspCNN_Enhanced,
                 n_cells: int = 25,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 8e-4,
                 gamma: float = 0.99,
                 freeze_backbone: bool = True):
        super().__init__()
        self.grasp_net = grasp_net
        self.n_cells = n_cells
        self.gamma = gamma
        self.q_head = LocalGridQHead(grasp_net.feature_size, n_cells=n_cells)
        self.q_head_target = copy.deepcopy(self.q_head)
        self.ce_loss = nn.CrossEntropyLoss()
        if freeze_backbone:
            for param in self.grasp_net.parameters():
                param.requires_grad = False
        trainable = list(self.q_head.parameters())
        if not freeze_backbone:
            trainable += [p for p in self.grasp_net.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            trainable, lr=learning_rate, weight_decay=weight_decay,
        )
        self._dqn_step = 0

    def to(self, device):
        super().to(device)
        return self

    def encode_features(self, states: torch.Tensor) -> torch.Tensor:
        was_training = self.grasp_net.training
        self.grasp_net.eval()
        try:
            grad = any(p.requires_grad for p in self.grasp_net.parameters())
            with torch.set_grad_enabled(grad):
                return self.grasp_net.encode_features(states)
        finally:
            if was_training:
                self.grasp_net.train()

    def q_values(self, states: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.encode_features(states))

    @torch.no_grad()
    def q_values_target(self, states: torch.Tensor) -> torch.Tensor:
        return self.q_head_target(self.encode_features(states))

    @torch.no_grad()
    def select_cell(self, states: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
        if epsilon > 0 and random.random() < epsilon:
            batch = states.shape[0]
            return torch.randint(0, self.n_cells, (batch,), device=states.device)
        return self.q_values(states).argmax(dim=1)

    def update_shaping(self, batch: Dict) -> Dict[str, float]:
        states = batch['states']
        labels = batch['cell_labels'].long()
        self.q_head.train()
        logits = self.q_values(states)
        loss = self.ce_loss(logits, labels)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.parameters() if p.requires_grad], 5.0,
        )
        self.optimizer.step()
        acc = (logits.argmax(dim=1) == labels).float().mean().item()
        return {
            'total': loss.item(),
            'ce': loss.item(),
            'acc': acc,
            'grad_norm': grad_norm.item(),
        }

    def update_dqn(self, batch: Dict) -> Dict[str, float]:
        states = batch['states']
        actions = batch['cell_actions'].long()
        rewards = batch['rewards'].view(-1, 1)
        next_states = batch['next_states']
        dones = batch['dones'].view(-1, 1)

        q = self.q_values(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q = self.q_values_target(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + (1.0 - dones) * self.gamma * next_q

        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.parameters() if p.requires_grad], 5.0,
        )
        self.optimizer.step()
        self._dqn_step += 1
        return {
            'total': loss.item(),
            'dqn': loss.item(),
            'grad_norm': grad_norm.item(),
        }

    def sync_target(self):
        self.q_head_target.load_state_dict(self.q_head.state_dict())

    def set_backbone_trainable(self, trainable: bool):
        for param in self.grasp_net.parameters():
            param.requires_grad = trainable
        lr = self.optimizer.param_groups[0]['lr']
        wd = self.optimizer.defaults.get('weight_decay', 8e-4)
        params = list(self.q_head.parameters())
        if trainable:
            params += list(self.grasp_net.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)

    def save_model(self, filepath: str, training_step: int = 0):
        torch.save({
            'grasp_net_state_dict': self.grasp_net.state_dict(),
            'q_head_state_dict': self.q_head.state_dict(),
            'q_head_target_state_dict': self.q_head_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': training_step,
            'dqn_step': self._dqn_step,
            'n_cells': self.n_cells,
        }, filepath)

    def load_model(self, filepath: str) -> int:
        checkpoint = torch.load(filepath, map_location=next(self.parameters()).device)
        self.grasp_net.load_state_dict(checkpoint['grasp_net_state_dict'], strict=False)
        self.q_head.load_state_dict(checkpoint['q_head_state_dict'])
        self.q_head_target.load_state_dict(checkpoint.get(
            'q_head_target_state_dict', checkpoint['q_head_state_dict'],
        ))
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f'   ↳ Grid optimizer not restored: {e}')
        self._dqn_step = int(checkpoint.get('dqn_step', 0))
        return int(checkpoint.get('training_step', 0))


def create_local_grid_module(grasp_net: UR3GraspCNN_Enhanced,
                             n_cells: int = 25,
                             learning_rate: float = 1e-4,
                             weight_decay: float = 8e-4,
                             gamma: float = 0.99,
                             freeze_backbone: bool = True) -> LocalGridModule:
    return LocalGridModule(
        grasp_net=grasp_net,
        n_cells=n_cells,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gamma=gamma,
        freeze_backbone=freeze_backbone,
    )
