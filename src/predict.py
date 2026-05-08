"""Prediction wrapper around the saved best model and vectorizer."""

from functools import lru_cache
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "vectorizer.pkl"


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[object, object]:
    model = joblib.load(BEST_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict_job_role(
    skills: str,
    experience_level: str = "",
    education_level: str = "",
) -> str:
    model, vectorizer = load_artifacts()
    parts = [part.strip() for part in (skills, experience_level, education_level) if part and part.strip()]
    combined_text = " ".join(parts)
    features = vectorizer.transform([combined_text])
    prediction = model.predict(features)[0]
    return str(prediction)
