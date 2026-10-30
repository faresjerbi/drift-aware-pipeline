
import unittest
import numpy as np

from drift_detector import cosine_similarity, THRESHOLD


def calculate_drift(baseline, recent):
    baseline_centroid = np.mean(baseline, axis=0)
    recent_centroid = np.mean(recent, axis=0)

    similarity = cosine_similarity(
        baseline_centroid,
        recent_centroid,
    )

    return 1.0 - similarity


def make_vectors_with_drift_score(score):
    """Create unit vectors whose cosine distance is approximately score."""
    similarity = 1.0 - score
    angle = np.arccos(similarity)

    baseline = np.array([[1.0, 0.0]])
    recent = np.array([[np.cos(angle), np.sin(angle)]])

    return baseline, recent


class TestDriftThreshold(unittest.TestCase):

    def test_identical_distributions_do_not_trigger_drift(self):
        baseline = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
        ])

        score = calculate_drift(baseline, baseline.copy())

        self.assertLessEqual(score, THRESHOLD)

    def test_orthogonal_distributions_trigger_drift(self):
        baseline = np.array([[1.0, 0.0]])
        recent = np.array([[0.0, 1.0]])

        score = calculate_drift(baseline, recent)

        self.assertGreater(score, THRESHOLD)

    def test_score_below_threshold_does_not_trigger_drift(self):
        baseline, recent = make_vectors_with_drift_score(
            THRESHOLD - 0.01
        )

        score = calculate_drift(baseline, recent)

        self.assertLessEqual(score, THRESHOLD)

    def test_score_above_threshold_triggers_drift(self):
        baseline, recent = make_vectors_with_drift_score(
            THRESHOLD + 0.01
        )

        score = calculate_drift(baseline, recent)

        self.assertGreater(score, THRESHOLD)

    def test_exact_threshold_does_not_trigger_drift(self):
        baseline, recent = make_vectors_with_drift_score(THRESHOLD)

        score = calculate_drift(baseline, recent)

        self.assertAlmostEqual(score, THRESHOLD, places=10)
        self.assertFalse(score > THRESHOLD)


if __name__ == "__main__":
    unittest.main(verbosity=2)
