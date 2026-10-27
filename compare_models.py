
import time

from chatbot import client, MODEL
from db import get_connection
from evaluate_model import TEST_CASES, evaluate_response

CANDIDATE_MODEL = "gpt-oss:120b"
MINIMUM_SCORE = 80.0


def evaluate_one_model(model_name):
    passed = 0
    total = len(TEST_CASES)
    successful_latencies = []

    print(f"\n=== Evaluating model: {model_name} ===")

    for test_case in TEST_CASES:
        print(f"\nTest: {test_case['name']}")
        start = time.perf_counter()

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer accurately and clearly. "
                            "Be concise but informative."
                        ),
                    },
                    {
                        "role": "user",
                        "content": test_case["prompt"],
                    },
                ],
            )

            elapsed = time.perf_counter() - start
            successful_latencies.append(elapsed)

            answer = response.choices[0].message.content or ""
            success, issues = evaluate_response(answer, test_case)

            if success:
                passed += 1
                print("Result: PASS")
            else:
                print("Result: FAIL")
                for issue in issues:
                    print(f" - {issue}")

            print(f"Latency: {elapsed:.2f} seconds")
            print(f"Response preview: {answer[:250]}")

        except Exception as error:
            print(f"Result: ERROR - {error}")

    score = passed / total * 100 if total else 0.0
    average_latency = (
        sum(successful_latencies) / len(successful_latencies)
        if successful_latencies
        else None
    )

    print(f"\nScore: {passed}/{total} ({score:.1f}%)")

    if average_latency is not None:
        print(f"Average latency: {average_latency:.2f} seconds")
    else:
        print("Average latency: unavailable")

    return {
        "model": model_name,
        "score": score,
        "average_latency": average_latency,
    }



def save_comparison(current, candidate):
    current_passes = current["score"] >= MINIMUM_SCORE
    candidate_passes = candidate["score"] >= MINIMUM_SCORE

    if candidate_passes and not current_passes:
        decision = "promote_candidate"
        details = (
            "Candidate passed the quality threshold while "
            "the current model failed. Recommendation only."
        )

    elif current_passes and not candidate_passes:
        decision = "keep_current"
        details = (
            "Current model passed the quality threshold while "
            "the candidate failed."
        )

    elif not current_passes and not candidate_passes:
        decision = "inconclusive"
        details = (
            "Neither model passed the quality threshold. "
            "Further evaluation is needed."
        )

    elif candidate["score"] > current["score"]:
        decision = "promote_candidate"
        details = "Candidate achieved a higher evaluation score."

    elif candidate["score"] < current["score"]:
        decision = "keep_current"
        details = "Current model achieved a higher evaluation score."

    elif (
        candidate["average_latency"] is not None
        and current["average_latency"] is not None
        and candidate["average_latency"] < current["average_latency"]
    ):
        decision = "promote_candidate"
        details = (
            "Scores were equal; candidate had lower average latency. "
            "Recommendation only, not a deployment."
        )

    else:
        decision = "inconclusive"
        details = "No clear candidate advantage was established."

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO public.model_comparison (
                    current_model,
                    candidate_model,
                    current_score,
                    candidate_score,
                    current_avg_latency_seconds,
                    candidate_avg_latency_seconds,
                    decision,
                    details
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    current["model"],
                    candidate["model"],
                    current["score"],
                    candidate["score"],
                    current["average_latency"],
                    candidate["average_latency"],
                    decision,
                    details,
                ),
            )

    print("\nComparison saved to Supabase.")
    print(f"Decision recommendation: {decision}")
    print(f"Details: {details}")
    print("No model was automatically deployed.")


def main():
    print("=== Model Comparison ===")
    print(f"Current model: {MODEL}")
    print(f"Candidate model: {CANDIDATE_MODEL}")

    current = evaluate_one_model(MODEL)
    candidate = evaluate_one_model(CANDIDATE_MODEL)

    print("\n=== Comparison Summary ===")
    print(f"Current score: {current['score']:.1f}%")
    print(f"Candidate score: {candidate['score']:.1f}%")

    save_comparison(current, candidate)


if __name__ == "__main__":
    main()
