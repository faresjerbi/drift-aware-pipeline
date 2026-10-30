
from db import get_connection

WINDOW_SIZE = 20

# Provisional alert thresholds, not production-validated.
MAX_REFUSAL_RATE = 0.20  # 20%
MAX_TOXICITY_RATE = 0.05  # 5%


def get_behavior_rates():
    query = """
        WITH ranked AS (
            SELECT
                id,
                refusal_flag,
                toxicity_flag,
                ROW_NUMBER() OVER (ORDER BY id DESC) AS rn
            FROM public.interaction_log
        ),
        windows AS (
            SELECT
                CASE
                    WHEN rn <= %s THEN 'recent'
                    ELSE 'previous'
                END AS period,
                refusal_flag,
                toxicity_flag
            FROM ranked
            WHERE rn <= %s
        )
        SELECT
            period,
            COUNT(*) AS interactions,
            COUNT(*) FILTER (
                WHERE refusal_flag IS TRUE
            ) AS refusals,
            COUNT(*) FILTER (
                WHERE toxicity_flag IS TRUE
            ) AS toxic_responses
        FROM windows
        GROUP BY period;
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (WINDOW_SIZE, WINDOW_SIZE * 2),
            )
            rows = cursor.fetchall()

    results = {}

    for period, total, refusals, toxic_responses in rows:
        results[period] = {
            "interactions": total,
            "refusal_rate": refusals / total if total else 0.0,
            "toxicity_rate": (
                toxic_responses / total if total else 0.0
            ),
        }

    return results


def main():
    results = get_behavior_rates()

    if "recent" not in results:
        print("Not enough interaction data to monitor behavior.")
        return

    recent = results["recent"]
    previous = results.get("previous")

    print("=== Behavior Monitor ===")
    print(f"Recent interactions: {recent['interactions']}")
    print(f"Refusal rate: {recent['refusal_rate']:.1%}")
    print(f"Toxicity rate: {recent['toxicity_rate']:.1%}")

    if previous:
        print("\n=== Previous Window ===")
        print(f"Interactions: {previous['interactions']}")
        print(f"Refusal rate: {previous['refusal_rate']:.1%}")
        print(f"Toxicity rate: {previous['toxicity_rate']:.1%}")

    alerts = []

    if recent["refusal_rate"] > MAX_REFUSAL_RATE:
        alerts.append("Refusal rate exceeded threshold.")

    if recent["toxicity_rate"] > MAX_TOXICITY_RATE:
        alerts.append("Toxicity rate exceeded threshold.")

    if alerts:
        for alert in alerts:
            print(f"ALERT: {alert}")
    else:
        print("\nNo threshold alerts.")

    # Compare rates with the previous window when available.
    if previous:
        refusal_change = (
            recent["refusal_rate"] - previous["refusal_rate"]
        )
        toxicity_change = (
            recent["toxicity_rate"] - previous["toxicity_rate"]
        )
        print(f"Refusal rate change: {refusal_change:+.1%}")
        print(f"Toxicity rate change: {toxicity_change:+.1%}")


if __name__ == "__main__":
    main()
