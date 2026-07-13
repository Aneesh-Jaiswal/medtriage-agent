"""
Intake Agent: extracts structured fields from free-text patient input using
an LLM with structured output (Pydantic schema via LangChain).
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class IntakeRecord(BaseModel):
    age: int = Field(description="Patient age in years, estimate if not stated")
    sex: str = Field(description="Patient sex, 'male'/'female'/'unknown'")
    symptom_text: str = Field(description="Normalized comma-separated list of symptoms")
    duration_hours: int = Field(description="How long symptoms have been present, in hours")
    urgency_flag: str = Field(description="'low', 'medium', or 'high' based on described symptoms")


INTAKE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a clinical intake assistant. Extract structured fields from "
            "the patient's free-text description. Be conservative: if symptoms "
            "suggest a medical emergency (e.g. chest pain, stroke signs, severe "
            "trauma, loss of consciousness), set urgency_flag to 'high'.",
        ),
        ("human", "{patient_text}"),
    ]
)


def build_intake_chain(model: str = "gpt-4o-mini"):
    llm = ChatOpenAI(model=model, temperature=0)
    structured_llm = llm.with_structured_output(IntakeRecord)
    return INTAKE_PROMPT | structured_llm


def run_intake(patient_text: str, model: str = "gpt-4o-mini") -> IntakeRecord:
    chain = build_intake_chain(model=model)
    return chain.invoke({"patient_text": patient_text})
