
import unittest

from metrics import (
    record_drift_event,
    record_accuracy_evaluation,
    record_pipeline_decision,
    set_active_behavior_alerts,
)


class TestPrometheusMetrics(unittest.TestCase):

    def test_record_drift_event(self):
        record_drift_event("embedding_drift", 0.25)

    def test_record_accuracy_evaluation(self):
        record_accuracy_evaluation("test-model", 90.0, False)

    def test_record_accuracy_drop(self):
        record_accuracy_evaluation("test-model", 75.0, True)

    def test_record_pipeline_decision(self):
        record_pipeline_decision("accepted")

    def test_set_active_behavior_alerts(self):
        set_active_behavior_alerts(2)


if __name__ == "__main__":
    unittest.main()
