"""
FastAPI app exposing the MedTriage-Agent pipeline as an HTTP API.

Run with:
    uvicorn api.main:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agents.graph import run_triage

app = FastAPI(
    title="MedTriage-Agent API",
    description=(
        "Agentic clinical intake, risk-triage, and prior-authorization "
        "decision system. Demo project using synthetic data only."
    ),
    version="0.1.0",
)


class TriageRequest(BaseModel):
    patient_text: str
    plan_id: str


class TriageResponse(BaseModel):
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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/triage", response_model=TriageResponse)
def triage(request: TriageRequest):
    try:
        result = run_triage(request.patient_text, request.plan_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result
