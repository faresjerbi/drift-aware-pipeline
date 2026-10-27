
from db import get_connection
from evaluate_model import evaluate_model

CANDIDATE_VERSION = "gpt-oss:20b-cloud-simulated-v2"
MINIMUM_EVALUATION_SCORE = 80.0


def retry_rejected_events():
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT drift_event_id
                FROM public.drift_event_processing
                WHERE status = 'rejected'
                ORDER BY drift_event_id
            """)
            event_ids = [row[0] for row in cursor.fetchall()]

    if not event_ids:
        print("No rejected events to retry.")
        return

    for event_id in event_ids:
        print(f"\nRetrying rejected drift event #{event_id}...")

        try:
            score = evaluate_model()
            status = (
                "accepted"
                if score >= MINIMUM_EVALUATION_SCORE
                else "rejected"
            )

            with get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute("""
                        UPDATE public.drift_event_processing
                        SET status = %s,
                            evaluation_score = %s,
                            candidate_version = %s,
                            processed_at = NOW(),
                            details = %s
                        WHERE drift_event_id = %s
                          AND status = 'rejected'
                    """, (
                        status,
                        score,
                        CANDIDATE_VERSION,
                        (
                            "Retry passed evaluation. Simulation only; "
                            "no model was trained or deployed."
                            if status == "accepted"
                            else "Retry did not meet the evaluation threshold."
                        ),
                        event_id,
                    ))

            print(f"Result: {status}")
            print(f"Evaluation score: {score:.1f}%")

        except Exception as error:
            print(f"Retry failed for event #{event_id}: {error}")


if __name__ == "__main__":
    retry_rejected_events()
