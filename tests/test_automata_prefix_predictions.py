import unittest

import numpy as np

from src.models.automata import ProbabilisticAutomata


class TestAutomataPrefixPredictions(unittest.TestCase):
    def test_prefix_labels_match_naive_prefix_loop_on_small_sequence(self):
        patterns = ["aaa", "aab", "abb", "abc", "bcc", "abc", "aab"]
        automata = ProbabilisticAutomata().fit(patterns)

        fast_preds = automata.predict_prefix_labels(patterns)
        naive_preds: list[int] = []
        for idx in range(1, len(patterns) + 1):
            decision = automata.predict_sequence(patterns[:idx])["decision"]
            naive_preds.append(1 if decision == "anomaly" else 0)

        np.testing.assert_array_equal(fast_preds, np.asarray(naive_preds, dtype=int))


if __name__ == "__main__":
    unittest.main()
