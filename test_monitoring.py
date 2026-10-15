
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


def test_prediction_report_rejects_empty_input():
    with pytest.raises(ValueError):
        generate_prediction_report([])
