
import unittest
from unittest.mock import patch, MagicMock

from accuracy_monitor import record_evaluation


class TestAccuracyMonitor(unittest.TestCase):

    @patch("accuracy_monitor.get_connection")
    def test_first_evaluation_has_no_baseline(self, mock_connection):
        conn = MagicMock()
        cursor = MagicMock()

        mock_connection.return_value.__enter__.return_value = conn
        conn.cursor.return_value.__enter__.return_value = cursor

        cursor.fetchone.side_effect = [(None,), (1,)]

        result = record_evaluation(
            model_name="test-model",
            evaluation_score=90.0,
            passed_tests=9,
            total_tests=10,
        )

        self.assertIsNone(result["baseline_score"])
        self.assertFalse(result["accuracy_drop_detected"])

    @patch("accuracy_monitor.get_connection")
    def test_drop_triggers_event(self, mock_connection):
        conn = MagicMock()
        cursor = MagicMock()

        mock_connection.return_value.__enter__.return_value = conn
        conn.cursor.return_value.__enter__.return_value = cursor

        cursor.fetchone.side_effect = [(95.0,), (2,)]

        result = record_evaluation(
            model_name="test-model",
            evaluation_score=80.0,
            passed_tests=8,
            total_tests=10,
        )

        self.assertTrue(result["accuracy_drop_detected"])
        self.assertEqual(result["drop_percentage_points"], 15.0)

    @patch("accuracy_monitor.get_connection")
    def test_small_drop_does_not_trigger_event(self, mock_connection):
        conn = MagicMock()
        cursor = MagicMock()

        mock_connection.return_value.__enter__.return_value = conn
        conn.cursor.return_value.__enter__.return_value = cursor

        cursor.fetchone.side_effect = [(95.0,), (2,)]

        result = record_evaluation(
            model_name="test-model",
            evaluation_score=90.0,
            passed_tests=9,
            total_tests=10,
        )

        self.assertFalse(result["accuracy_drop_detected"])


if __name__ == "__main__":
    unittest.main()
