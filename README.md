# MedTriage-Agent

**Agentic clinical intake, risk-triage, and prior-authorization decision system.**

MedTriage-Agent simulates a real-world healthcare workflow: a patient submits
symptoms and insurance information, and a multi-agent graph (built with
LangGraph + LangChain) routes the case through intake parsing, eligibility
verification, ML-based risk scoring (PyTorch), and a prior-authorization
decision — with human-in-the-loop escalation for edge cases.

This project is built for demonstration/portfolio purposes using **synthetic
data only**. No real patient information (PHI) is used anywhere in this
repository.

---

## Why this project

Healthcare payers and providers spend enormous operational effort on two
things: **triaging patient urgency** and **deciding prior-authorization
requests** against policy rules. This project models both as a single
agentic pipeline:

1. **Intake Agent** — extracts structured fields (symptoms, duration,
   severity, demographics) from free-text patient input using an LLM.
2. **Eligibility Agent** — performs RAG retrieval over a synthetic
   insurance-policy knowledge base to check plan coverage rules.
3. **Risk Classifier** — a PyTorch neural network trained on synthetic
   clinical features to output an urgency/risk score (low/medium/high).
4. **Prior-Auth Agent** — combines eligibility + risk output to reach an
   approve / deny / escalate decision, with a natural-language rationale.
5. **Human-in-the-loop node** — LangGraph conditional edge that routes
   ambiguous or high-risk cases to a "pending human review" state instead
   of auto-deciding.

---

## Architecture

```
                        ┌─────────────────────┐
                        │   Patient Input      │
                        │ (symptoms + plan id) │
                        └──────────┬───────────┘
                                   ▼
                        ┌─────────────────────┐
                        │   Intake Agent       │  (LangChain LLM chain)
                        └──────────┬───────────┘
                                   ▼
                 ┌─────────────────────────────────┐
                 │        Eligibility Agent          │  (RAG over policy docs)
                 └──────────────┬─────────────────────┘
                                   ▼
                 ┌─────────────────────────────────┐
                 │      Risk Classifier (PyTorch)    │
                 └──────────────┬─────────────────────┘
                                   ▼
                     ┌───────────────────────┐
                     │ conditional routing    │
                     │ (LangGraph edge)       │
                     └──────┬─────────┬────────┘
                            ▼         ▼
                 ┌────────────────┐ ┌────────────────────┐
                 │ Prior-Auth Agent│ │ Human Review Queue  │
                 │ (approve/deny)  │ │ (ambiguous/high-risk)│
                 └────────────────┘ └────────────────────┘
```

---

## Tech stack

| Layer            | Tool                              |
|-------------------|-----------------------------------|
| Agent orchestration | LangGraph                       |
| LLM chains / RAG  | LangChain                         |
| ML risk model     | PyTorch                           |
| Vector store       | ChromaDB                         |
| API                | FastAPI                          |
| Data               | Synthetic (Faker-generated)       |

---

## Project structure

```
medtriage-agent/
├── agents/
│   ├── graph.py            # LangGraph StateGraph definition & routing
│   ├── intake_agent.py      # Structured extraction from patient text
│   ├── eligibility_agent.py # RAG-based coverage check
│   └── prior_auth_agent.py  # Final decision + rationale
├── models/
│   ├── risk_classifier.py   # PyTorch model definition
│   ├── train.py             # Training script on synthetic data
│   └── checkpoints/         # Saved model weights (gitignored)
├── rag/
│   ├── policy_docs/         # Synthetic insurance policy text files
│   └── retriever.py         # Chroma vector store setup
├── data/synthetic/
│   └── generate_patients.py # Synthetic patient record generator
├── api/
│   └── main.py               # FastAPI app exposing /triage endpoint
├── notebooks/
│   └── explore_risk_model.ipynb
├── tests/
│   └── test_graph.py
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Setup

```bash
git clone https://github.com/<your-username>/medtriage-agent.git
cd medtriage-agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY or ANTHROPIC_API_KEY
```

### 1. Generate synthetic data
```bash
python data/synthetic/generate_patients.py
```

### 2. Train the risk classifier
```bash
python -m models.train --epochs 30
```

### 3. Build the policy vector store
```bash
python -m rag.retriever --build
```

### 4. Run the API
```bash
uvicorn api.main:app --reload
```

### 5. Try it
```bash
curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"patient_text": "45 year old male, chest pain and shortness of breath for 2 hours", "plan_id": "PLAN_GOLD_002"}'
```

---

## Roadmap / extension ideas

- [ ] Swap the synthetic risk classifier for a fine-tuned BioClinicalBERT model
- [ ] Add a React front-end intake form + decision trace viewer
- [ ] Add LangSmith tracing for full agent observability
- [ ] Add eval harness comparing agent decisions against labeled synthetic ground truth
- [ ] Containerize with Docker + docker-compose for one-command startup

---

## Disclaimer

This is a portfolio/demo project. It uses **only synthetic data**, is **not
a certified clinical decision support tool**, and should not be used to make
real healthcare or insurance decisions.

## License

MIT

---

## How it works

The system processes each patient request through a multi-step AI workflow. First, the Intake Agent extracts structured information such as symptoms, duration, and demographics from the patient's free-text input. Next, the Eligibility Agent retrieves relevant insurance policy information using RAG to determine coverage and authorization requirements. The extracted clinical features are then passed to a PyTorch risk classifier, which predicts the patient's urgency level (low, medium, or high). Finally, LangGraph orchestrates the workflow by combining the insurance eligibility and risk assessment to either generate a prior-authorization decision (approve/deny) or route high-risk or ambiguous cases to a human review step, simulating a real-world healthcare decision-making pipeline.
