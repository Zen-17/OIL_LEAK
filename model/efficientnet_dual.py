"""
EfficientNet-B0 dual-head classifier for GPR B-scan anomaly detection.
Uses torchvision built-in backbone (no extra dependencies).

  Head A  binary anomaly score  P ∈ [0, 1]     sigmoid
  Head B  5-class target type   logits (5,)     cross-entropy

  Loss = 0.5 × BCELoss(A) + 0.5 × CrossEntropyLoss(B)

Note: EfficientNet-B0 (torchvision) shares the same 1280-dim feature space and
MBConv structure as EfficientNet-Lite0. SE modules are present in B0 but do not
affect RKNN INT8 quantisation; swap backbone to Lite0 via timm if needed later.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Head B class order — must match CLUTTER_TO_TYPE in train.py
CLASS_NAMES = ["oil", "metal_pipe", "nonmetal_pipe", "rock", "background"]
N_TYPES     = len(CLASS_NAMES)   # 5
FEAT_DIM    = 1280


class EfficientNetDual(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base    = efficientnet_b0(weights=weights)

        # keep conv layers + pooling, drop the original classifier
        self.features  = base.features       # (B, 1280, 7, 7)
        self.pool      = base.avgpool        # AdaptiveAvgPool2d(1) → (B, 1280, 1, 1)
        self.drop_feat = base.classifier[0]  # Dropout(0.2) from pretrained
        # LayerNorm normalises the 1280-dim vector per sample, preventing the
        # linear heads from developing extreme weights that explode val loss.
        self.feat_norm = nn.LayerNorm(FEAT_DIM)
        self.drop_head = nn.Dropout(p=0.4)   # extra regularisation before heads

        self.head_a = nn.Linear(FEAT_DIM, 1)
        self.head_b = nn.Linear(FEAT_DIM, N_TYPES)

    def forward(self, x: torch.Tensor):
        """
        x        : (B, 3, 224, 224)
        logit_a  : (B,)    raw anomaly logit (apply sigmoid for probability)
        logits_b : (B, 5)  type logits
        """
        feat     = self.features(x)               # (B, 1280, 7, 7)
        feat     = self.pool(feat).flatten(1)      # (B, 1280)
        feat     = self.drop_feat(feat)
        feat     = self.feat_norm(feat)            # unit-scale features
        feat     = self.drop_head(feat)
        logit_a  = self.head_a(feat).squeeze(1)   # raw logit; sigmoid applied at inference
        logits_b = self.head_b(feat)
        return logit_a, logits_b

    # ── Parameter group helpers ───────────────────────────────────────────────

    def freeze_backbone(self, ratio: float = 0.75):
        """Freeze first `ratio` of backbone blocks — Phase-1 training.
        ratio=1.0 freezes the entire backbone (use EfficientNet as fixed extractor).
        Operates on child modules so each block's weight+bias are frozen together.
        """
        children = list(self.features.children())   # stem + 7 MBConv blocks + head conv
        n_freeze = int(len(children) * ratio)
        for i, block in enumerate(children):
            for p in block.parameters():
                p.requires_grad = i >= n_freeze
        # feat_norm and heads are always trainable regardless of ratio
        always_train = (list(self.feat_norm.parameters())
                        + list(self.head_a.parameters())
                        + list(self.head_b.parameters()))
        for p in always_train:
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def trainable_params(self) -> list:
        return [p for p in self.parameters() if p.requires_grad]


# ── Loss ──────────────────────────────────────────────────────────────────────

def dual_loss(
    logit_a:   torch.Tensor,          # (B,)   float  raw logit (no sigmoid)
    logits_b:  torch.Tensor,          # (B, 5) float
    label_a:   torch.Tensor,          # (B,)   float  {0.0, 1.0}
    label_b:   torch.Tensor,          # (B,)   long   {0..4}
    w_a: float = 0.5,
    w_b: float = 0.5,
    weight_b:  torch.Tensor = None,   # (5,) per-class weight for Head B
    label_smoothing: float = 0.1,     # prevents overconfident predictions
) -> torch.Tensor:
    if label_smoothing > 0:
        label_a = label_a * (1 - label_smoothing) + 0.5 * label_smoothing
    loss_a = F.binary_cross_entropy_with_logits(logit_a, label_a)
    loss_b = F.cross_entropy(logits_b, label_b, weight=weight_b,
                             label_smoothing=label_smoothing)
    return w_a * loss_a + w_b * loss_b
