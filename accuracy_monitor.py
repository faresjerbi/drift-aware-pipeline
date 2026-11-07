
from db import get_connection
from metrics import record_accuracy_evaluation

# A drop of 10 percentage points or more triggers an alert.
ACCURACY_DROP_THRESHOLD = 10.0


def record_evaluation(
    model_name: str,
    evaluation_score: float,
    passed_tests: int,
    total_tests: int,
    details: dict | None = None,
) -> dict:
    """Save an evaluation and detect a significant accuracy drop."""

    if not model_name.strip():
        raise ValueError("model_name cannot be empty")

    if total_tests <= 0:
        raise ValueError("total_tests must be greater than zero")

    if not 0 <= evaluation_score <= 100:
        raise ValueError("evaluation_score must be between 0 and 100")

    if not 0 <= passed_tests <= total_tests:
        raise ValueError("passed_tests must be between 0 and total_tests")

    details = details or {}

    with get_connection() as conn:
        with conn.cursor() as cursor:
            # Get the most recent evaluation before recording this one.
            cursor.execute(
                """
                SELECT evaluation_score
                FROM public.evaluation_runs
                WHERE model_name = %s
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (model_name,),
            )
            previous = cursor.fetchone()

            baseline_score = (
                float(previous[0])
                if previous and previous[0] is not None
                else None
            )

            # Save the new evaluation.
            cursor.execute(
                """
                INSERT INTO public.evaluation_runs (
                    model_name,
                    evaluation_score,
                    passed_tests,
                    total_tests,
                    details
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    model_name,
                    evaluation_score,
                    passed_tests,
                    total_tests,
                    __import__("json").dumps(details),
                ),
            )
            evaluation_id = cursor.fetchone()[0]

            drop = (
                baseline_score - evaluation_score
                if baseline_score is not None
                else 0.0
            )

            accuracy_drop_detected = (
                baseline_score is not None
                and drop >= ACCURACY_DROP_THRESHOLD
            )

            if accuracy_drop_detected:
                cursor.execute(
                    """
                    INSERT INTO public.drift_events (
                        signal,
                        score,
                        threshold,
                        action_taken
                    )
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        "accuracy_drop",
                        drop,
                        ACCURACY_DROP_THRESHOLD,
                        "detected",
                    ),
                )

    # Update Prometheus metrics for every evaluation.
    record_accuracy_evaluation(
        model_name,
        evaluation_score,
        accuracy_drop_detected,
    )

    result = {
        "evaluation_id": evaluation_id,
        "model_name": model_name,
        "evaluation_score": evaluation_score,
        "baseline_score": baseline_score,
        "drop_percentage_points": round(drop, 2),
        "accuracy_drop_detected": accuracy_drop_detected,
    }

    print("\n=== Accuracy Monitor ===")
    print(f"Model: {model_name}")
    print(f"Evaluation score: {evaluation_score:.1f}%")

    if baseline_score is None:
        print("Baseline: none (first evaluation)")
    else:
        print(f"Previous score: {baseline_score:.1f}%")
        print(f"Score drop: {drop:.1f} percentage points")

    print(
        "Accuracy drop detected: "
        f"{'YES' if accuracy_drop_detected else 'NO'}"
    )

    if accuracy_drop_detected:
        print("An accuracy_drop event was saved to Supabase.")

    return result


def evaluate_and_record():
    """Run model evaluation and persist the result."""
    from evaluate_model import evaluate_model

    result = evaluate_model()

    record_evaluation(
        model_name=result["model_name"],
        evaluation_score=result["score"],
        passed_tests=result["passed_tests"],
        total_tests=result["total_tests"],
        details={
            "source": "evaluate_model.py",
        },
    )

    return result
