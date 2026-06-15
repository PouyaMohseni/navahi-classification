"""
Multi-task model from the Navahi paper:

  Input: MERT embeddings (2304-dim)
  Shared backbone: 4 FC layers with ReLU, reducing to 32 dims
                   Dropout(0.2) after layer 3, Dropout(0.4) after layer 4
  Classification head: FC(32 → num_classes)
  Regression head:     FC(32 → 2)  [normalized lat, lon]

Loss: L_total = L_cls + lambda_reg * L_reg
  L_cls = CrossEntropyLoss
  L_reg = MSELoss
  lambda_reg = 1.0  (paper ratio 2.5:1 cls:reg, λ=1)
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FEATURE_DIM, NUM_CLASSES, LAMBDA_REG


class NavahiClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        num_classes: int = NUM_CLASSES,
        lambda_reg: float = LAMBDA_REG,
    ):
        super().__init__()
        self.lambda_reg = lambda_reg

        # Shared backbone: 4 FC layers progressively reducing to 32
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.cls_head = nn.Linear(32, num_classes)
        self.reg_head = nn.Linear(32, 2)

        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.reg_loss_fn = nn.MSELoss()

    def forward(self, x):
        h = self.backbone(x)
        logits = self.cls_head(h)
        coords = self.reg_head(h)
        return logits, coords

    def compute_loss(self, logits, coords_pred, labels, coords_true):
        l_cls = self.cls_loss_fn(logits, labels)
        l_reg = self.reg_loss_fn(coords_pred, coords_true)
        total = l_cls + self.lambda_reg * l_reg
        return total, l_cls.item(), l_reg.item()
