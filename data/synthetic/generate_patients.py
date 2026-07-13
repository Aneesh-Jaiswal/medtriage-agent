"""
Generates synthetic patient intake records for training the risk classifier
and for demoing the MedTriage-Agent pipeline end-to-end.

No real patient data is used anywhere -- all records are randomly generated.
"""
import csv
import json
import random
from pathlib import Path

random.seed(42)

OUTPUT_DIR = Path(__file__).parent
SYMPTOM_BANK = [
    ("chest pain", 0.9),
    ("shortness of breath", 0.85),
    ("severe headache", 0.6),
    ("mild headache", 0.15),
    ("fever", 0.4),
    ("cough", 0.2),
    ("abdominal pain", 0.5),
    ("back pain", 0.2),
    ("dizziness", 0.45),
    ("fatigue", 0.15),
    ("rash", 0.1),
    ("sore throat", 0.1),
    ("nausea", 0.3),
    ("high blood pressure reading", 0.55),
    ("laceration requiring stitches", 0.5),
    ("broken bone suspected", 0.75),
    ("allergic reaction", 0.6),
    ("routine checkup", 0.02),
    ("prescription refill", 0.02),
    ("follow-up visit", 0.05),
]

PLANS = ["PLAN_GOLD_001", "PLAN_GOLD_002", "PLAN_SILVER_001", "PLAN_BRONZE_001", "PLAN_MEDICARE_ADV"]


def make_record(patient_id: int) -> dict:
    age = random.randint(1, 95)
    sex = random.choice(["male", "female"])
    n_symptoms = random.choice([1, 1, 2, 2, 3])
    chosen = random.sample(SYMPTOM_BANK, n_symptoms)
    symptom_text = ", ".join(s[0] for s in chosen)
    base_risk = sum(s[1] for s in chosen) / len(chosen)

    # Age adjustment: very young/old patients skew slightly higher risk
    age_factor = 0.15 if age < 5 or age > 70 else 0.0
    duration_hours = random.choice([1, 2, 4, 8, 24, 48, 72, 168])
    duration_factor = 0.1 if duration_hours <= 4 else -0.05

    risk_score = min(1.0, max(0.0, base_risk + age_factor + duration_factor + random.uniform(-0.1, 0.1)))

    if risk_score >= 0.65:
        label = "high"
    elif risk_score >= 0.35:
        label = "medium"
    else:
        label = "low"

    return {
        "patient_id": f"P{patient_id:05d}",
        "age": age,
        "sex": sex,
        "symptom_text": symptom_text,
        "duration_hours": duration_hours,
        "plan_id": random.choice(PLANS),
        "risk_score": round(risk_score, 3),
        "risk_label": label,
    }


def main(n_records: int = 2000):
    records = [make_record(i) for i in range(n_records)]

    csv_path = OUTPUT_DIR / "patients.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    json_path = OUTPUT_DIR / "patients.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Generated {n_records} synthetic patient records.")
    print(f"  -> {csv_path}")
    print(f"  -> {json_path}")


if __name__ == "__main__":
    main()
