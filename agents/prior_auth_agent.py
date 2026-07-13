"""
Prior-Auth Agent: combines the risk classifier output and eligibility
decision into a final approve / deny / escalate decision with a
natural-language rationale.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class PriorAuthDecision(BaseModel):
    decision: str = Field(description="'approve', 'deny', or 'escalate'")
    rationale: str = Field(description="Short explanation for the decision")


DECISION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a prior-authorization decision assistant. Rules:\n"
            "- If risk_label is 'high', always 'approve' (never deny high-risk care).\n"
            "- If prior_auth_required is False, always 'approve'.\n"
            "- If risk_label is 'medium' and prior_auth_required is True, 'escalate' "
            "to a human reviewer unless the eligibility rationale clearly supports approval.\n"
            "- If risk_label is 'low' and prior_auth_required is True, lean toward 'deny' "
            "or redirect to lower-acuity care, unless there is a clear clinical reason not to.\n"
            "Always explain your reasoning in one or two sentences.",
        ),
        (
            "human",
            "Risk label: {risk_label}\n"
            "Prior authorization required: {prior_auth_required}\n"
            "Eligibility rationale: {eligibility_rationale}",
        ),
    ]
)


def run_prior_auth_decision(
    risk_label: str, prior_auth_required: bool, eligibility_rationale: str
) -> PriorAuthDecision:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(PriorAuthDecision)
    chain = DECISION_PROMPT | structured_llm

    return chain.invoke(
        {
            "risk_label": risk_label,
            "prior_auth_required": prior_auth_required,
            "eligibility_rationale": eligibility_rationale,
        }
    )
