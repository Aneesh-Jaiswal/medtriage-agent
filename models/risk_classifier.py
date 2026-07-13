"""
PyTorch risk/urgency classifier.

Takes structured features derived from patient intake (age, duration,
a bag-of-symptom-keywords vector, etc.) and predicts a triage risk class:
low / medium / high.

This is intentionally a small, fast feed-forward network -- the point of
this project is the agentic orchestration around it, not SOTA modeling.
Swap in a fine-tuned transformer (e.g. BioClinicalBERT) here for a more
advanced version.
"""
from __future__ import annotations

import torch
import torch.nn as nn

RISK_LABELS = ["low", "medium", "high"]


class RiskClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_label(self, x: torch.Tensor) -> str:
        self.eval()
        logits = self.forward(x)
        idx = torch.argmax(logits, dim=-1).item()
        return RISK_LABELS[idx]


def load_model(checkpoint_path: str, input_dim: int) -> RiskClassifier:
    model = RiskClassifier(input_dim=input_dim)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
