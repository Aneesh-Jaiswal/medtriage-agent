"""
Eligibility Agent: given a patient's plan_id and risk label, retrieves the
relevant policy chunks and asks an LLM whether prior authorization is
required, citing the policy language it relied on.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from rag.retriever import get_retriever


class EligibilityDecision(BaseModel):
    prior_auth_required: bool = Field(description="Whether prior authorization is required")
    rationale: str = Field(description="Short explanation citing the relevant policy language")


ELIGIBILITY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an insurance eligibility assistant. Given policy excerpts and "
            "a patient's risk level, determine whether prior authorization is "
            "required for their case. Base your answer strictly on the provided "
            "policy text.",
        ),
        (
            "human",
            "Plan ID: {plan_id}\nRisk label: {risk_label}\n\n"
            "Relevant policy excerpts:\n{policy_context}",
        ),
    ]
)


def run_eligibility_check(plan_id: str, risk_label: str) -> EligibilityDecision:
    retriever = get_retriever(k=3)
    query = f"{plan_id} prior authorization requirements for {risk_label} risk cases"
    docs = retriever.invoke(query)
    policy_context = "\n---\n".join(d.page_content for d in docs)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(EligibilityDecision)
    chain = ELIGIBILITY_PROMPT | structured_llm

    return chain.invoke(
        {
            "plan_id": plan_id,
            "risk_label": risk_label,
            "policy_context": policy_context,
        }
    )
