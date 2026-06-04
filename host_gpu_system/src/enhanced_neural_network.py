#!/usr/bin/env python3
"""
Enhanced Neural Network Architecture for UR3 Grasping System

Utilizes a MobileNetV2 backbone for efficient RGB-D feature extraction.
Using behavior cloning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
import numpy as np
from typing import Tuple, Optional, Dict, List
import logging


class UR3GraspCNN_Enhanced(nn.Module):
    """
    UR3 Grasping Network using a pretrained MobileNetV2 backbone.
    Adapted for 4-channel input (RGB-D) to support depth-aware feature extraction.
    """

    def __init__(self,
                 input_channels: int = 4,
                 input_size: Tuple[int, int] = (224, 224),
                 num_grasp_classes: int = 4,
                 output_6dof: bool = True,
                 use_attention: bool = True):
        super(UR3GraspCNN_Enhanced, self).__init__()

        self.input_channels    = input_channels
        self.input_size        = input_size
        self.num_grasp_classes = num_grasp_classes
        self.output_6dof       = output_6dof
        self.use_attention     = use_attention

        # =========================================================================
        # BACKBONE INITIALIZATION
        # =========================================================================
        weights    = MobileNet_V2_Weights.DEFAULT
        mobilenet  = models.mobilenet_v2(weights=weights)
        self.backbone = mobilenet.features

        # Adapt initial convolutional layer to accept 4-channel RGB-D input
        self._modify_first_layer(input_channels)

        # Standard MobileNetV2 output channel dimension
        self.feature_size = 1280

        if self.use_attention:
            self.attention = SpatialAttention(self.feature_size)

        # =========================================================================
        # OUTPUT HEADS
        # =========================================================================

        self.grasp_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_grasp_classes)
        )

        if self.output_6dof:
            self.pose_regressor = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.feature_size, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 6)  # Coordinates: x, y, z, rx, ry, rz
            )

        # Quality estimation retained for monitoring; gradients inactive in BC mode
        self.quality_predictor = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

        self.aux_position_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)  # Global spatial coordinates: (object_X, object_Z)
        )

        self._initialize_head_weights()

    def _modify_first_layer(self, input_channels):
        """Reconstructs the first convolutional layer to support extended channels."""
        if input_channels == 3:
            return
            
        original_layer = self.backbone[0][0]
        new_layer = nn.Conv2d(
            in_channels=input_channels,
            out_channels=original_layer.out_channels,
            kernel_size=original_layer.kernel_size,
            stride=original_layer.stride,
            padding=original_layer.padding,
            bias=original_layer.bias
        )
        
        with torch.no_grad():
            new_layer.weight[:, :3, :, :] = original_layer.weight
            if input_channels > 3:
                new_layer.weight[:, 3:, :, :] = torch.mean(
                    original_layer.weight, dim=1, keepdim=True
                )
        self.backbone[0][0] = new_layer

    def _initialize_head_weights(self):
        """Applies standard normal initialization to all fully connected heads."""
        heads = [self.grasp_classifier, self.quality_predictor, self.aux_position_head]
        if self.output_6dof:
            heads.append(self.pose_regressor)
            
        for m in heads:
            if isinstance(m, nn.Module):
                for layer in m.modules():
                    if isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, 0, 0.01)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        if self.use_attention:
            features = self.attention(features)

        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)

        outputs = {}
        outputs['grasp_class']  = self.grasp_classifier(features)
        if self.output_6dof:
            outputs['pose_6dof'] = self.pose_regressor(features)
        outputs['quality']      = self.quality_predictor(features)
        outputs['aux_position'] = self.aux_position_head(features)

        return outputs


class SpatialAttention(nn.Module):
    """Refines feature maps by emphasizing structurally significant spatial regions."""
    def __init__(self, in_channels: int):
        super(SpatialAttention, self).__init__()
        self.conv1   = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1, bias=False)
        self.bn1     = nn.BatchNorm2d(in_channels // 16)
        self.conv2   = nn.Conv2d(in_channels // 16, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = F.relu(self.bn1(self.conv1(x)))
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)
        return x * attn


class BehaviorCloningModule(nn.Module):
    """
    Supervised learning wrapper managing the grasping network updates.
    Integrates pose regression and auxiliary positioning supervision.
    """

    def __init__(self,
                 grasp_net: UR3GraspCNN_Enhanced,
                 learning_rate: float = 5e-4,
                 weight_decay: float = 8e-4):
        super(BehaviorCloningModule, self).__init__()

        self.grasp_net = grasp_net

        self.optimizer = torch.optim.Adam(
            self.grasp_net.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.regression_loss     = nn.SmoothL1Loss()
        self.classification_loss = nn.CrossEntropyLoss()

        # =========================================================================
        # LOSS WEIGHT CONFIGURATION
        # =========================================================================
        self.pose_loss_weight  = 1.0  # primary target: 6-DOF pose regression
        self.aux_loss_weight   = 0.5  # secondary target: object position bounding
        self.grasp_loss_weight = 0.0  # disabled: tracked strictly for monitoring

    def to(self, device):
        super().to(device)
        return self

    def update_networks(self, batch: Dict) -> Dict[str, float]:
        """
        Executes a single gradient update step based on a batch of teacher demonstrations.

        Expected Batch Tensors (Mapped to Device):
          states              [B, 4, 224, 224]
          pose_labels         [B, 6]   -> teacher 6-DOF pose targets
          grasp_labels        [B]      -> grasp class integer
          rewards             [B, 1]   -> success mask for pose loss (1.0 = train, 0.0 = skip)
          aux_position_labels [B, 2]   -> spatial coordinates (X, Z)
        """
        states              = batch['states']
        pose_labels         = batch['pose_labels']
        grasp_labels        = batch['grasp_labels']
        rewards             = batch['rewards']
        aux_position_labels = batch.get('aux_position_labels', None)

        self.grasp_net.train()

        outputs       = self.grasp_net(states)
        pose_preds    = outputs['pose_6dof']    
        grasp_logits  = outputs['grasp_class']  
        aux_pos_preds = outputs['aux_position'] 

        # Target Masking
        # Height (Y - index 1) is masked as it remains structurally constant across the platform.
        # Indices: 0=x, 1=y (MASKED), 2=z, 3=rx, 4=ry, 5=rz
        POSE_MASK = torch.tensor([10, 0, 10, 0, 0, 1],
                                 dtype=torch.float32,
                                 device=pose_preds.device)

        # Success-only BC: imitate grasp poses only from successful demonstrations
        success_mask = (rewards >= 0.95).squeeze(1)

        pose_loss = torch.tensor(0.0, device=pose_preds.device)
        if success_mask.any():
            pose_loss = self.regression_loss(
                pose_preds[success_mask]  * POSE_MASK,
                pose_labels[success_mask] * POSE_MASK
            )

        # Auxiliary loss conditioning
        aux_loss = torch.tensor(0.0, device=pose_preds.device)
        if aux_position_labels is not None:
            valid_mask = (aux_position_labels.abs().sum(dim=1) > 0.001)
            if valid_mask.any():
                aux_loss = self.regression_loss(
                    aux_pos_preds[valid_mask],
                    aux_position_labels[valid_mask]
                )

        grasp_loss = self.classification_loss(grasp_logits, grasp_labels)

        total_loss = (
            self.grasp_loss_weight * grasp_loss +
            self.pose_loss_weight  * pose_loss  +
            self.aux_loss_weight   * aux_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.grasp_net.parameters(), 5.0)
        self.optimizer.step()

        return {
            'total':     total_loss.item(),
            'pose':      pose_loss.item(),
            'grasp':     grasp_loss.item(),
            'aux':       aux_loss.item(),
            'grad_norm': grad_norm.item()
        }

    def save_model(self, filepath: str):
        torch.save({
            'model_state_dict':     self.grasp_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath)
        self.grasp_net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"   ↳ Optimizer state not restored (architecture changed): {e}")


class ImageProcessor:
    """Prepares and normalizes real-time image data for tensor processing."""
    
    def __init__(self,
                 device: torch.device = None,
                 image_size: Tuple[int, int] = (224, 224),
                 is_simulation: bool = False,
                 crop_y0: float = 0.215,
                 crop_y1: float = 0.584,
                 crop_x0: float = 0.39,
                 crop_x1: float = 0.5565):
                     
        self.device        = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size    = image_size
        self.is_simulation = is_simulation

        # =========================================================================
        # VISION PREPROCESSING & CROP CONFIGURATION
        # =========================================================================
        # Defines proportional bounding boxes (0.0 to 1.0).
        # These can be visually verified via the debug frames stored by the GPU server.
        self.crop_y0 = crop_y0
        self.crop_y1 = crop_y1
        self.crop_x0 = crop_x0
        self.crop_x1 = crop_x1

    def process_rgbd_image(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> torch.Tensor:
        """Converts raw RGB-D arrays into normalized, cropped torch tensors."""
        h, w = rgb_image.shape[:2]
        y0 = int(self.crop_y0 * h)
        y1 = int(self.crop_y1 * h)
        x0 = int(self.crop_x0 * w)
        x1 = int(self.crop_x1 * w)

        rgb_image   = rgb_image[y0:y1, x0:x1]
        depth_image = depth_image[y0:y1, x0:x1]

        DEPTH_MIN = 0.50
        DEPTH_MAX = 1.00
        depth_clipped    = np.clip(depth_image, DEPTH_MIN, DEPTH_MAX)
        depth_normalized = (depth_clipped - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)

        # Convert 8-bit RGB to normalized float32 prior to depth concatenation
        rgb_normalized = rgb_image.astype(np.float32) / 255.0

        rgbd_image  = np.concatenate([rgb_normalized, depth_normalized[:, :, np.newaxis]], axis=2)
        rgbd_tensor = torch.from_numpy(rgbd_image.transpose(2, 0, 1)).float()

        rgbd_tensor = F.interpolate(
            rgbd_tensor.unsqueeze(0), size=self.image_size, mode='bilinear', align_corners=False
        )

        mean = torch.tensor([0.485, 0.456, 0.406, 0.5], device=rgbd_tensor.device).view(1, 4, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225, 0.5], device=rgbd_tensor.device).view(1, 4, 1, 1)
        rgbd_tensor = (rgbd_tensor - mean) / std

        return rgbd_tensor.to(self.device)


def create_model(config: Dict) -> Tuple[UR3GraspCNN_Enhanced, BehaviorCloningModule, 'ImageProcessor']:
    """Instantiates the complete architectural pipeline."""
    grasp_net = UR3GraspCNN_Enhanced(
        input_channels=config.get('input_channels', 4),
        input_size=tuple(config.get('input_size', [224, 224])),
        num_grasp_classes=config.get('num_grasp_classes', 4),
        output_6dof=config.get('output_6dof', True),
        use_attention=config.get('use_attention', True)
    )

    bc_module = BehaviorCloningModule(
        grasp_net=grasp_net,
        learning_rate=config.get('learning_rate', 5e-4),
        weight_decay=config.get('weight_decay', 8e-4)
    )

    pretrained_path = config.get('pretrained_weights')
    if pretrained_path:
        import os
        if os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            bc_module.load_model(pretrained_path)

    image_processor = ImageProcessor(
        image_size=tuple(config.get('input_size', [224, 224])),
        is_simulation=config.get('is_simulation', False),
        crop_y0=config.get('crop_y0', 0.215),
        crop_y1=config.get('crop_y1', 0.584),
        crop_x0=config.get('crop_x0', 0.39),
        crop_x1=config.get('crop_x1', 0.5565)
    )

    return grasp_net, bc_module, image_processor