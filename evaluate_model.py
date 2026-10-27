
from chatbot import client, MODEL


TEST_CASES = [
    {
        "name": "Python functions",
        "prompt": "What is a Python function? Explain its purpose.",
        "required_concepts": [
            ["function"],
            ["reusable", "reusability", "use multiple times", "used again"],
        ],
        "min_words": 20,
    },
    {
        "name": "Primary keys",
        "prompt": "What is a primary key in a relational database?",
        "required_concepts": [
            ["unique", "uniquely identifies", "uniqueness"],
            ["row", "record"],
        ],
        "min_words": 25,
    },
    {
        "name": "Data drift",
        "prompt": "What is data drift in machine learning?",
        "required_concepts": [
            ["data"],
            ["change", "shift", "differ", "difference"],
            ["model", "machine learning"],
        ],
        "min_words": 25,
    },
]


def evaluate_response(response_text, test_case):
    text = response_text.lower().strip()
    words = text.split()

    if not text:
        return False, ["Empty response"]

    issues = []

    for alternatives in test_case["required_concepts"]:
        if not any(term in text for term in alternatives):
            issues.append(
                "Missing concept (accepted alternatives: "
                + ", ".join(alternatives)
                + ")"
            )

    if len(words) < test_case["min_words"]:
        issues.append(
            f"Too short: {len(words)} words; "
            f"minimum is {test_case['min_words']}"
        )

    return len(issues) == 0, issues


def evaluate_model():
    passed = 0
    total = len(TEST_CASES)

    print(f"\n=== Evaluating {MODEL} ===\n")

    for test_case in TEST_CASES:
        print(f"Test: {test_case['name']}")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer accurately and clearly. "
                            "Be concise but informative."
                        ),
                    },
                    {"role": "user", "content": test_case["prompt"]},
                ],
            )

            answer = response.choices[0].message.content or ""
            success, issues = evaluate_response(answer, test_case)

            if success:
                passed += 1
                print("Result: PASS")
            else:
                print("Result: FAIL")
                for issue in issues:
                    print(f" - {issue}")

            print(f"Response: {answer[:500]}\n")

        except Exception as error:
            print(f"Result: ERROR - {error}\n")

    score = passed / total * 100
    print("=== Evaluation Summary ===")
    print(f"Passed: {passed}/{total}")
    print(f"Evaluation score: {score:.1f}%")

    return score


if __name__ == "__main__":
    evaluate_model()
