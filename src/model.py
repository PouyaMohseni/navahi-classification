"""
Multi-task classifier from the Navahi paper (mlptest/mlptest-reg2.py SimpleMLP).

Architecture:
  input_dim (27648)
  → Linear → BatchNorm1d → ReLU                         (1024)
  → Linear → BatchNorm1d → ReLU → Dropout(0.2)          (256)
  → Linear → BatchNorm1d → ReLU → Dropout(0.4)          (128)
  → Linear → ReLU                                        (32)
  ┌─ cls_head: Linear(32 → num_classes)
  └─ reg_head: Linear(32 → 2)

Loss: L_total = cls_weight * CrossEntropy + lambda_reg * MSE
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
        input_dim:   int   = FEATURE_DIM,
        num_classes: int   = NUM_CLASSES,
        lambda_reg:  float = LAMBDA_REG,
        cls_weight:  float = 1.0,
    ):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.cls_weight = cls_weight

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 32),
            nn.ReLU(),
        )

        self.cls_head = nn.Linear(32, num_classes)
        self.reg_head = nn.Linear(32, 2)

        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.reg_loss_fn = nn.MSELoss()

    def forward(self, x):
        h = self.shared(x)
        return self.cls_head(h), self.reg_head(h)

    def compute_loss(self, logits, coords_pred, labels, coords_true):
        l_cls = self.cls_loss_fn(logits, labels)
        l_reg = self.reg_loss_fn(coords_pred, coords_true)
        total = self.cls_weight * l_cls + self.lambda_reg * l_reg
        return total, l_cls.item(), l_reg.item()
