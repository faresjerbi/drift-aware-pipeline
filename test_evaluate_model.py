
import unittest

from evaluate_model import evaluate_response


class TestEvaluateResponse(unittest.TestCase):
    def setUp(self):
        self.test_case = {
            "name": "Python functions",
            "required_concepts": [
                ["function"],
                ["reusable", "reusability", "use multiple times", "used again"],
            ],
            "min_words": 10,
        }

    def test_valid_response_passes(self):
        response = (
            "A Python function is a reusable block of code. "
            "You can call the function multiple times to perform "
            "a task without repeating the same code."
        )
        passed, issues = evaluate_response(response, self.test_case)
        self.assertTrue(passed, issues)

    def test_empty_response_fails(self):
        passed, issues = evaluate_response("", self.test_case)
        self.assertFalse(passed)
        self.assertIn("Empty response", issues)

    def test_missing_required_concept_fails(self):
        response = (
            "A Python function is a block of code that performs "
            "a task. It helps organize a program into smaller parts."
        )
        passed, issues = evaluate_response(response, self.test_case)
        self.assertFalse(passed)
        self.assertTrue(any("Missing concept" in issue for issue in issues))

    def test_short_response_fails(self):
        passed, issues = evaluate_response(
            "A function is reusable.",
            self.test_case,
        )
        self.assertFalse(passed)
        self.assertTrue(any("Too short" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
