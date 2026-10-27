
from datetime import datetime, timezone

from db import get_connection
from evaluate_model import evaluate_model

CANDIDATE_VERSION = "gpt-oss:20b-cloud-simulated-v2"
MINIMUM_EVALUATION_SCORE = 80.0


def get_unprocessed_events():
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT d.id, d.signal, d.score, d.threshold, d.action_taken
                FROM public.drift_events AS d
                LEFT JOIN public.drift_event_processing AS p
                    ON p.drift_event_id = d.id
                WHERE p.drift_event_id IS NULL
                ORDER BY d.timestamp ASC, d.id ASC
            """)
            return cursor.fetchall()


def record_event(
    event_id, status, candidate_version=None,
    evaluation_score=None, details=None
):
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO public.drift_event_processing
                    (drift_event_id, status, candidate_version,
                     evaluation_score, processed_at, details)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (drift_event_id) DO NOTHING
            """, (
                event_id, status, candidate_version,
                evaluation_score, datetime.now(timezone.utc), details
            ))


def record_candidate(version, trigger_reason, score):
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO public.model_versions
                    (version, date, trigger_reason, eval_score)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (version) DO NOTHING
            """, (
                version, datetime.now(timezone.utc),
                trigger_reason, score
            ))


def process_event(event):
    event_id, signal, score, threshold, action = event

    print(f"\n--- Drift event #{event_id} ---")
    print(f"Signal: {signal}")
    print(f"Score: {score:.4f}; threshold: {threshold:.4f}")
    print(f"Action: {action}")

    if action != "drift_detected":
        record_event(
            event_id, "skipped",
            details="Event did not indicate drift."
        )
        print("Skipped: no drift detected.")
        return

    try:
        evaluation_score = evaluate_model()

        if evaluation_score < MINIMUM_EVALUATION_SCORE:
            record_event(
                event_id, "rejected",
                candidate_version=CANDIDATE_VERSION,
                evaluation_score=evaluation_score,
                details="Candidate failed the evaluation threshold."
            )
            print("Candidate rejected.")
            return

        reason = (
            f"Simulation only; drift event #{event_id}; "
            f"signal={signal}; drift_score={score:.4f}"
        )

        record_candidate(
            CANDIDATE_VERSION, reason, evaluation_score
        )
        record_event(
            event_id, "accepted",
            candidate_version=CANDIDATE_VERSION,
            evaluation_score=evaluation_score,
            details=(
                "Candidate passed evaluation. Simulation only; "
                "no model was trained or deployed."
            )
        )

        print("Candidate passed evaluation and was recorded.")

    except Exception as error:
        record_event(
            event_id, "failed",
            candidate_version=CANDIDATE_VERSION,
            details=str(error)[:1000]
        )
        print(f"Processing failed: {error}")


def main():
    print("=== Drift Processing and Retraining Simulation ===")

    events = get_unprocessed_events()

    if not events:
        print("No unprocessed drift events.")
        return

    print(f"Unprocessed events: {len(events)}")

    for event in events:
        process_event(event)

    print("\n=== Processing Summary ===")

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT status, COUNT(*)
                FROM public.drift_event_processing
                GROUP BY status
                ORDER BY status
            """)
            for status, count in cursor.fetchall():
                print(f"{status}: {count}")


if __name__ == "__main__":
    main()
