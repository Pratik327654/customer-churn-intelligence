
from datetime import datetime, timezone

import pandas as pd
import os


def generate_prediction_report(probabilities, threshold=0.40):
    """Summarize churn probabilities and predicted risk levels."""

    probabilities = pd.Series(probabilities, dtype="float64")

    if probabilities.empty:
        raise ValueError("At least one prediction probability is required.")

    if probabilities.isna().any():
        raise ValueError("Probabilities cannot contain missing values.")

    if not probabilities.between(0, 1).all():
        raise ValueError("Probabilities must be between 0 and 1.")

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    high_risk = probabilities >= threshold

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_customers": int(len(probabilities)),
        "high_risk_customers": int(high_risk.sum()),
        "low_risk_customers": int((~high_risk).sum()),
        "average_churn_probability": round(float(probabilities.mean()), 4),
        "threshold": float(threshold),
    }
    
HISTORY_FILE = "prediction_history.csv"


def save_prediction(probability, threshold=0.40):
    """Append one prediction to the local CSV history."""

    report = generate_prediction_report(
        [probability],
        threshold=threshold
    )

    row = pd.DataFrame([{
        "timestamp_utc": report["generated_at_utc"],
        "churn_probability": float(probability),
        "risk_level": (
            "High Risk" if probability >= threshold else "Low Risk"
        ),
        "threshold": float(threshold),
    }])

    file_exists = os.path.exists(HISTORY_FILE)

    row.to_csv(
        HISTORY_FILE,
        mode="a",
        header=not file_exists,
        index=False
    )

    return row
