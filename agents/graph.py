"""
LangGraph orchestrator: wires the Intake, Eligibility, Risk Classifier, and
Prior-Auth agents into a single stateful graph with conditional routing to
a human-review node for ambiguous / high-risk-but-uncovered cases.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, TypedDict

import torch
from langgraph.graph import StateGraph, END

from agents.intake_agent import run_intake
from agents.eligibility_agent import run_eligibility_check
from agents.prior_auth_agent import run_prior_auth_decision
from models.features import featurize, feature_dim
from models.risk_classifier import load_model

CHECKPOINT_PATH = Path(__file__).parent.parent / "models" / "checkpoints" / "risk_classifier.pt"


class TriageState(TypedDict, total=False):
    patient_text: str
    plan_id: str
    age: int
    sex: str
    symptom_text: str
    duration_hours: int
    risk_label: str
    prior_auth_required: bool
    eligibility_rationale: str
    decision: str
    decision_rationale: str
    needs_human_review: bool


def intake_node(state: TriageState) -> TriageState:
    record = run_intake(state["patient_text"])
    return {
        "age": record.age,
        "sex": record.sex,
        "symptom_text": record.symptom_text,
        "duration_hours": record.duration_hours,
    }


def risk_node(state: TriageState) -> TriageState:
    x = featurize(state["age"], state["duration_hours"], state["symptom_text"])
    if CHECKPOINT_PATH.exists():
        model = load_model(str(CHECKPOINT_PATH), input_dim=feature_dim())
        label = model.predict_label(x.unsqueeze(0))
    else:
        # Fallback heuristic if no trained checkpoint is available yet.
        label = "high" if "chest pain" in state["symptom_text"].lower() else "medium"
    return {"risk_label": label}


def eligibility_node(state: TriageState) -> TriageState:
    result = run_eligibility_check(state["plan_id"], state["risk_label"])
    return {
        "prior_auth_required": result.prior_auth_required,
        "eligibility_rationale": result.rationale,
    }


def prior_auth_node(state: TriageState) -> TriageState:
    result = run_prior_auth_decision(
        state["risk_label"], state["prior_auth_required"], state["eligibility_rationale"]
    )
    needs_review = result.decision == "escalate"
    return {
        "decision": result.decision,
        "decision_rationale": result.rationale,
        "needs_human_review": needs_review,
    }


def human_review_node(state: TriageState) -> TriageState:
    # Placeholder for a real human-in-the-loop queue (e.g. write to a DB /
    # ticketing system). Here we just mark the case as pending.
    return {"decision": "pending_human_review"}


def route_after_decision(state: TriageState) -> str:
    return "human_review" if state.get("needs_human_review") else END


def build_graph():
    graph = StateGraph(TriageState)

    graph.add_node("intake", intake_node)
    graph.add_node("risk", risk_node)
    graph.add_node("eligibility", eligibility_node)
    graph.add_node("prior_auth", prior_auth_node)
    graph.add_node("human_review", human_review_node)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "risk")
    graph.add_edge("risk", "eligibility")
    graph.add_edge("eligibility", "prior_auth")
    graph.add_conditional_edges(
        "prior_auth",
        route_after_decision,
        {"human_review": "human_review", END: END},
    )
    graph.add_edge("human_review", END)

    return graph.compile()


def run_triage(patient_text: str, plan_id: str) -> TriageState:
    app = build_graph()
    result = app.invoke({"patient_text": patient_text, "plan_id": plan_id})
    return result
