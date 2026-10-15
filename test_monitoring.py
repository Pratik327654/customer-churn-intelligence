
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



