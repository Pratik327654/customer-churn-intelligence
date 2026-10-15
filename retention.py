
def get_retention_actions(contract, tenure, tech_support, online_security):
    actions = []

    if contract == "Month-to-month":
        actions.append("Offer a suitable long-term contract incentive")

    if tenure <= 12:
        actions.append("Provide early-customer onboarding support")

    if tech_support == "No":
        actions.append("Offer a technical support review")

    if online_security == "No":
        actions.append("Explain available online security options")

    if not actions:
        actions.append("Offer a routine satisfaction check-in")

    return actions
