"""
Shared feature engineering for the risk classifier: turns a raw patient
record (age, symptom text, duration) into a fixed-size numeric feature
vector that the PyTorch model can consume.
"""
from __future__ import annotations

import torch

# Fixed vocabulary of symptom keywords -> keeps the feature vector fixed size
# and interpretable. In a production system this would be replaced by
# embeddings from a clinical language model.
SYMPTOM_VOCAB = [
    "chest pain", "shortness of breath", "severe headache", "mild headache",
    "fever", "cough", "abdominal pain", "back pain", "dizziness", "fatigue",
    "rash", "sore throat", "nausea", "high blood pressure reading",
    "laceration requiring stitches", "broken bone suspected",
    "allergic reaction", "routine checkup", "prescription refill",
    "follow-up visit",
]

RISK_LABEL_TO_IDX = {"low": 0, "medium": 1, "high": 2}


def symptom_vector(symptom_text: str) -> list[float]:
    text = symptom_text.lower()
    return [1.0 if kw in text else 0.0 for kw in SYMPTOM_VOCAB]


def featurize(age: int, duration_hours: int, symptom_text: str) -> torch.Tensor:
    age_norm = age / 100.0
    duration_norm = min(duration_hours, 168) / 168.0
    vec = [age_norm, duration_norm] + symptom_vector(symptom_text)
    return torch.tensor(vec, dtype=torch.float32)


def feature_dim() -> int:
    return 2 + len(SYMPTOM_VOCAB)
