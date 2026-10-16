
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


def load_prediction_history():
    """Load saved prediction history if available."""

    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "churn_probability",
                "risk_level",
                "threshold",
            ]
        )

    return pd.read_csv(HISTORY_FILE)


def detect_numeric_drift(reference_stats, new_data, columns, threshold=0.20):
    """Detect numeric mean shifts relative to reference standard deviation."""

    results = []

    for col in columns:
        reference_mean = float(reference_stats.loc["mean", col])
        reference_std = float(reference_stats.loc["std", col])
        new_mean = float(new_data[col].mean())

        if pd.isna(reference_std) or reference_std == 0:
            drift_score = 0.0
        else:
            drift_score = abs(new_mean - reference_mean) / reference_std

        results.append({
            "feature": col,
            "reference_mean": round(reference_mean, 2),
            "new_mean": round(new_mean, 2),
            "drift_score": round(float(drift_score), 3),
            "drift_detected": drift_score > threshold,
        })

    return pd.DataFrame(results)


