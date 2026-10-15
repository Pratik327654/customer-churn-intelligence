

   
from retention import get_retention_actions


def test_month_to_month_customer():
    actions = get_retention_actions(
        "Month-to-month", 24, "Yes", "Yes"
    )
    assert "Offer a suitable long-term contract incentive" in actions


def test_new_customer():
    actions = get_retention_actions(
        "Two year", 6, "Yes", "Yes"
    )
    assert "Provide early-customer onboarding support" in actions


def test_technical_support_recommendation():
    actions = get_retention_actions(
        "One year", 24, "No", "Yes"
    )
    assert "Offer a technical support review" in actions


def test_default_recommendation():
    actions = get_retention_actions(
        "Two year", 24, "Yes", "Yes"
    )
    assert actions == ["Offer a routine satisfaction check-in"]
