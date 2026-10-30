
from behavior_monitor import (
    get_behavior_rates,
    MAX_REFUSAL_RATE,
    MAX_TOXICITY_RATE,
)
from db import get_connection

def check_behavior_alerts():
    """Return alerts for elevated refusal or toxicity rates."""
    results = get_behavior_rates()
    recent = results.get("recent")

    if not recent:
        return {
            "status": "insufficient_data",
            "alerts": [],
            "details": "No recent interactions available.",
        }

    if recent["interactions"] < 20:
        return {
            "status": "insufficient_data",
            "alerts": [],
            "details": (
                f"Only {recent['interactions']} recent interactions; "
                "at least 20 are required."
            ),
        }

    alerts = []

    if recent["refusal_rate"] > MAX_REFUSAL_RATE:
        alerts.append(
            f"Refusal rate elevated: "
            f"{recent['refusal_rate']:.1%}"
        )

    if recent["toxicity_rate"] > MAX_TOXICITY_RATE:
        alerts.append(
            f"Toxicity rate elevated: "
            f"{recent['toxicity_rate']:.1%}"
        )

    return {
        "status": "alert" if alerts else "normal",
        "alerts": alerts,
        "details": (
            f"Analyzed {recent['interactions']} recent interactions."
        ),
    }


if __name__ == "__main__":
    result = check_behavior_alerts()
    print("=== Behavior Alert Check ===")
    print(f"Status: {result['status']}")
    print(result["details"])

    for alert in result["alerts"]:
        print(f"ALERT: {alert}")
