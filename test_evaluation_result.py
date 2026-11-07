
import unittest
from unittest.mock import patch

from evaluate_model import evaluate_model, TEST_CASES


MOCK_ANSWERS = {
    "Python functions": (
        "A Python function is a reusable block of code that performs "
        "a specific task. It helps organize programs and lets developers "
        "use the same logic multiple times without repeating code."
    ),
    "Primary keys": (
        "A primary key is a column or set of columns that uniquely "
        "identifies each row in a relational database table. It must "
        "contain unique values and cannot be null."
    ),
    "Data drift": (
        "Data drift occurs when the statistical properties or distribution "
        "of input data change over time. This change can affect a machine "
        "learning model because its production data may differ from "
        "the data used during training."
    ),
}


class TestEvaluationResult(unittest.TestCase):

    @patch("evaluate_model.client.chat.completions.create")
    def test_evaluator_returns_score_and_test_counts(self, mock_create):
        mock_create.side_effect = [
            type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message",
                                    (),
                                    {"content": MOCK_ANSWERS[test["name"]]},
                                )()
                            },
                        )()
                    ]
                },
            )()
            for test in TEST_CASES
        ]

        result = evaluate_model()

        self.assertIsInstance(result, dict)
        self.assertIn("score", result)
        self.assertIn("passed_tests", result)
        self.assertEqual(result["total_tests"], len(TEST_CASES))
        self.assertEqual(result["passed_tests"], len(TEST_CASES))
        self.assertEqual(result["score"], 100.0)


if __name__ == "__main__":
    unittest.main()
