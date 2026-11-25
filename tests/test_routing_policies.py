import numpy as np
import unittest

from src.routing.policies import compute_routing_scores


class RoutingPolicyTests(unittest.TestCase):
    def test_entropy_router_uses_base_two_units(self):
        probabilities = np.array([0.5, 0.25, 0.75])

        scores = compute_routing_scores(
            "R3_entropy",
            uncal_probs=probabilities,
        )

        expected = np.array([1.0, 0.8112781244591328, 0.8112781244591328])
        np.testing.assert_allclose(scores, expected)

    def test_entropy_router_prefers_calibrated_probabilities_when_available(self):
        uncalibrated = np.array([0.5])
        calibrated = np.array([0.1])

        scores = compute_routing_scores(
            "R3_entropy",
            uncal_probs=uncalibrated,
            cal_probs=calibrated,
        )

        np.testing.assert_allclose(scores, np.array([0.4689955935892812]))


if __name__ == "__main__":
    unittest.main()
