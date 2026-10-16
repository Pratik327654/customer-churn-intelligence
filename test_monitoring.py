
import pytest

from monitoring import generate_prediction_report


def test_prediction_report_counts_risk_levels():
    report = generate_prediction_report([0.2, 0.4, 0.8], threshold=0.4)

    assert report["total_customers"] == 3
    assert report["high_risk_customers"] == 2
    assert report["low_risk_customers"] == 1


def test_prediction_report_average_probability():
    report = generate_prediction_report([0.2, 0.4, 0.8])

    assert report["average_churn_probability"] == pytest.approx(0.4667)


def test_prediction_report_rejects_invalid_probability():
    with pytest.raises(ValueError):
        generate_prediction_report([0.2, 1.5])


def test_save_prediction_appends_to_csv(tmp_path, monkeypatch):
    import pandas as pd
    import monitoring

    history_file = tmp_path / "prediction_history.csv"
    monkeypatch.setattr(monitoring, "HISTORY_FILE", str(history_file))

    monitoring.save_prediction(0.75, threshold=0.40)
    monitoring.save_prediction(0.20, threshold=0.40)

    history = pd.read_csv(history_file)

    assert len(history) == 2
    assert list(history["risk_level"]) == ["High Risk", "Low Risk"]
    assert list(history["churn_probability"]) == [0.75, 0.20]


def test_detect_numeric_drift_flags_shifted_feature():
    import pandas as pd
    from monitoring import detect_numeric_drift

    reference_stats = pd.DataFrame({
        "tenure": [32.37, 24.56],
        "MonthlyCharges": [64.76, 30.09],
        "TotalCharges": [2283.30, 2266.77],
    }, index=["mean", "std"])

    new_data = pd.DataFrame({
        "tenure": [50, 55, 60],
        "MonthlyCharges": [60, 65, 70],
        "TotalCharges": [2000, 2200, 2400],
    })

    report = detect_numeric_drift(
        reference_stats,
        new_data,
        ["tenure", "MonthlyCharges", "TotalCharges"],
        threshold=0.20,
    )

    tenure_result = report.loc[report["feature"] == "tenure"].iloc[0]
    assert tenure_result["drift_detected"] == True




