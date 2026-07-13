"""
Tests that don't require any LLM API calls -- covers feature engineering
and the PyTorch model, which can run fully offline in CI.
"""
import torch

from models.features import featurize, feature_dim, symptom_vector
from models.risk_classifier import RiskClassifier, RISK_LABELS


def test_feature_dim_matches_vector_length():
    x = featurize(age=45, duration_hours=2, symptom_text="chest pain")
    assert x.shape[0] == feature_dim()


def test_symptom_vector_flags_known_keywords():
    vec = symptom_vector("Patient reports chest pain and fever")
    assert sum(vec) == 2  # chest pain + fever


def test_symptom_vector_ignores_unknown_terms():
    vec = symptom_vector("patient feels generally unwell")
    assert sum(vec) == 0


def test_risk_classifier_forward_shape():
    model = RiskClassifier(input_dim=feature_dim())
    x = torch.rand(4, feature_dim())
    logits = model(x)
    assert logits.shape == (4, len(RISK_LABELS))


def test_risk_classifier_predict_label_is_valid():
    model = RiskClassifier(input_dim=feature_dim())
    x = torch.rand(1, feature_dim())
    label = model.predict_label(x)
    assert label in RISK_LABELS
