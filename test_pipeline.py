
import unittest
from unittest.mock import patch
import numpy as np

from drift_detector import cosine_similarity
import pipeline


NORMAL_BEHAVIOR = {
    "status": "normal",
    "alerts": [],
    "details": "Analyzed 20 recent interactions.",
}


class TestDriftAwarePipeline(unittest.TestCase):

    @patch("pipeline.check_behavior_alerts", return_value=NORMAL_BEHAVIOR)
    @patch("pipeline.get_unprocessed_event")
    @patch("pipeline.detect_drift")
    @patch("pipeline.record_decision")
    @patch("pipeline.evaluate_one_model")
    def test_no_drift_skips_model_evaluation(
        self,
        mock_evaluate,
        mock_record,
        mock_detect,
        mock_get_event,
        mock_behavior,
    ):
        mock_get_event.return_value = (
            10, "embedding_drift", "no_drift", 0.08, 0.15
        )

        with patch("builtins.print"):
            pipeline.run_pipeline()

        mock_record.assert_called_once_with(
            10,
            "skipped",
            "No embedding drift detected. Model comparison was not needed.",
        )
        mock_evaluate.assert_not_called()

    @patch("pipeline.check_behavior_alerts", return_value=NORMAL_BEHAVIOR)
    @patch("pipeline.get_unprocessed_event")
    @patch("pipeline.detect_drift")
    @patch("pipeline.record_decision")
    @patch("pipeline.evaluate_one_model")
    @patch("pipeline.save_comparison")
    def test_drift_evaluates_both_models(
        self,
        mock_save,
        mock_evaluate,
        mock_record,
        mock_detect,
        mock_get_event,
        mock_behavior,
    ):
        mock_get_event.side_effect = [
            None,
            (11, "embedding_drift", "drift_detected", 0.22, 0.15),
        ]

        mock_evaluate.side_effect = [
            {"score": 70.0, "average_latency": 3.0},
            {"score": 90.0, "average_latency": 2.0},
        ]

        with patch("pipeline.MINIMUM_SCORE", 80), \
             patch("builtins.print"):
            pipeline.run_pipeline()

        self.assertEqual(mock_evaluate.call_count, 2)
        mock_save.assert_called_once()

        mock_record.assert_any_call(
            11,
            "processing",
            "Drift detected. Evaluating current and candidate models.",
        )

    @patch("pipeline.check_behavior_alerts", return_value=NORMAL_BEHAVIOR)
    @patch("pipeline.get_unprocessed_event")
    @patch("pipeline.detect_drift")
    @patch("pipeline.record_decision")
    @patch("pipeline.evaluate_one_model")
    def test_accuracy_drop_is_recorded_for_review(
        self,
        mock_evaluate,
        mock_record,
        mock_detect,
        mock_get_event,
        mock_behavior,
    ):
        mock_get_event.return_value = (
            12, "accuracy_drop", "detected", 15.0, 10.0
        )

        with patch("builtins.print"):
            pipeline.run_pipeline()

        mock_record.assert_called_once_with(
            12,
            "detected",
            (
                "Accuracy degradation detected. "
                "Evaluation history should be reviewed."
            ),
        )
        mock_evaluate.assert_not_called()

    @patch("pipeline.check_behavior_alerts", return_value=NORMAL_BEHAVIOR)
    @patch("pipeline.get_unprocessed_event")
    @patch("pipeline.detect_drift")
    @patch("pipeline.record_decision")
    def test_no_event_stops_pipeline(
        self,
        mock_record,
        mock_detect,
        mock_get_event,
        mock_behavior,
    ):
        mock_get_event.return_value = None
        mock_detect.return_value = None

        with patch("builtins.print"):
            pipeline.run_pipeline()

        mock_record.assert_not_called()


class TestDriftMath(unittest.TestCase):

    def test_identical_vectors_have_zero_drift(self):
        baseline = np.array([1.0, 0.0, 0.0])
        recent = np.array([1.0, 0.0, 0.0])

        similarity = cosine_similarity(baseline, recent)
        drift_score = 1.0 - similarity

        self.assertAlmostEqual(drift_score, 0.0, places=6)

    def test_orthogonal_vectors_have_high_drift(self):
        baseline = np.array([1.0, 0.0, 0.0])
        recent = np.array([0.0, 1.0, 0.0])

        similarity = cosine_similarity(baseline, recent)
        drift_score = 1.0 - similarity

        self.assertAlmostEqual(drift_score, 1.0, places=6)

    def test_zero_vector_similarity_is_safe(self):
        baseline = np.array([0.0, 0.0, 0.0])
        recent = np.array([1.0, 0.0, 0.0])

        self.assertEqual(cosine_similarity(baseline, recent), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
