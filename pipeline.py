
import sys
import time

from behavior_alerts import check_behavior_alerts
from db import get_connection
from accuracy_monitor import evaluate_and_record
from drift_detector import detect_drift
from compare_models import (
    CANDIDATE_MODEL,
    MINIMUM_SCORE,
    evaluate_one_model,
    save_comparison,
)
from chatbot import MODEL
from metrics import (
    start_metrics_server,
    record_pipeline_decision,
)

POLL_INTERVAL_SECONDS = 60


def get_unprocessed_event():
    """Return the oldest drift event without a processing record."""
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT d.id, d.signal, d.action_taken, d.score, d.threshold
                FROM public.drift_events AS d
                LEFT JOIN public.drift_event_processing AS p
                    ON p.drift_event_id = d.id
                WHERE p.drift_event_id IS NULL
                ORDER BY d.id ASC
                LIMIT 1
                """
            )
            return cursor.fetchone()


def record_decision(event_id, status, details):
    """Persist a pipeline decision and update its Prometheus counter."""
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO public.drift_event_processing
                    (drift_event_id, status, details, processed_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (drift_event_id)
                DO UPDATE SET
                    status = EXCLUDED.status,
                    details = EXCLUDED.details,
                    processed_at = NOW()
                """,
                (event_id, status, details),
            )

    record_pipeline_decision(status)


def run_pipeline():
    print("\n=== Drift-Aware Pipeline ===")

    # Check behavioral signals before processing drift events.
    behavior = check_behavior_alerts()

    print("\n=== Behavior Check ===")
    print(f"Status: {behavior['status']}")
    print(behavior["details"])

    for alert in behavior["alerts"]:
        print(f"ALERT: {alert}")

    # First, look for existing unprocessed events.
    event = get_unprocessed_event()

    if event is None:
        print("No unprocessed drift events found.")
        print("Checking for a new drift window...")

        detect_drift()

        # Check again because the detector may have created an event.
        event = get_unprocessed_event()

    if event is None:
        print("No new event needs processing in this cycle.")
        return

    event_id, signal, action, score, threshold = event

    print(f"Processing event #{event_id}")
    print(f"Drift result: {action}")
    print(f"Drift score: {score:.4f}")
    print(f"Threshold: {threshold:.4f}")

    # Route events according to their signal and action.
    if action == "no_drift":
        record_decision(
            event_id,
            "skipped",
            "No embedding drift detected. Model comparison was not needed.",
        )
        print("Event recorded as skipped.")
        return

    if signal == "accuracy_drop":
        record_decision(
            event_id,
            "detected",
            (
                "Accuracy degradation detected. "
                "Evaluation history should be reviewed."
            ),
        )
        print("Accuracy-drop event recorded for review.")
        return

    if action != "drift_detected":
        record_decision(
            event_id,
            "skipped",
            f"Unsupported event: signal={signal}, action={action}",
        )
        print("Unsupported event recorded as skipped.")
        return

    record_decision(
        event_id,
        "processing",
        "Drift detected. Evaluating current and candidate models.",
    )

    try:
        current = evaluate_one_model(MODEL)
        candidate = evaluate_one_model(CANDIDATE_MODEL)

        save_comparison(current, candidate)

        current_passes = current["score"] >= MINIMUM_SCORE
        candidate_passes = candidate["score"] >= MINIMUM_SCORE

        candidate_is_better = (
            candidate["score"] > current["score"]
            or (
                candidate["score"] == current["score"]
                and candidate["average_latency"] is not None
                and current["average_latency"] is not None
                and candidate["average_latency"]
                < current["average_latency"]
            )
        )

        if candidate_passes and (
            not current_passes or candidate_is_better
        ):
            status = "accepted"
            details = (
                f"Candidate {CANDIDATE_MODEL} passed evaluation. "
                "Promotion is simulated only; no model was deployed."
            )
        elif current_passes:
            status = "rejected"
            details = (
                "Candidate was not preferred. "
                "Current model remains active."
            )
        else:
            status = "rejected"
            details = (
                "Neither model met the minimum quality threshold. "
                "No model was deployed."
            )

        # Save the final decision.
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE public.drift_event_processing
                    SET status = %s,
                        candidate_version = %s,
                        evaluation_score = %s,
                        details = %s,
                        processed_at = NOW()
                    WHERE drift_event_id = %s
                    """,
                    (
                        status,
                        CANDIDATE_MODEL,
                        candidate["score"],
                        details,
                        event_id,
                    ),
                )

        # Update Prometheus only after the database update succeeds.
        record_pipeline_decision(status)

        print("\n=== Pipeline Summary ===")
        print(f"Event: #{event_id}")
        print(f"Current model score: {current['score']:.1f}%")
        print(f"Candidate model score: {candidate['score']:.1f}%")
        print(f"Processing status: {status}")
        print(details)

    except Exception as error:
        record_decision(
            event_id,
            "failed",
            f"Pipeline error: {error}",
        )
        raise


def run_continuous_monitoring():
    """Run the pipeline continuously and expose Prometheus metrics."""
    start_metrics_server(port=8000)

    print("\n=== Continuous Monitoring Started ===")
    print("Prometheus metrics: http://localhost:8000/metrics")
    print(f"Monitoring interval: {POLL_INTERVAL_SECONDS} seconds")
    print("Press Ctrl+C to stop.")

    while True:
        try:
            run_pipeline()
        except Exception as error:
            print(f"\nPipeline execution failed: {error}")

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check-accuracy":
        result = evaluate_and_record()

        print("\n=== Accuracy Evaluation Saved ===")
        print(f"Model: {result['model_name']}")
        print(f"Score: {result['score']:.1f}%")
        print(
            f"Passed: {result['passed_tests']}/"
            f"{result['total_tests']}"
        )

    elif len(sys.argv) > 1 and sys.argv[1] == "--serve":
        try:
            run_continuous_monitoring()
        except KeyboardInterrupt:
            print("\nContinuous monitoring stopped.")

    else:
        run_pipeline()
