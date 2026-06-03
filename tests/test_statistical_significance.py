import unittest

from src.evaluation.statistical_tests import calculate_statistical_significance


class TestStatisticalSignificance(unittest.TestCase):
    def test_wilcoxon_uses_paired_folds_per_seed(self):
        skab_results = [
            {
                "seed": 42,
                "original": {
                    "automata": {"raw": [{"f1": 0.4}, {"f1": 0.5}, {"f1": 0.6}]},
                    "lstm": {"raw": [{"f1": 0.7}, {"f1": 0.8}, {"f1": 0.9}]},
                    "gru": {"raw": [{"f1": 0.65}, {"f1": 0.75}, {"f1": 0.85}]},
                },
            }
        ]

        sig = calculate_statistical_significance(skab_results)
        self.assertEqual(sig["paired_fold_count_lstm"], 3)
        self.assertIn("automata_vs_lstm", sig)
        self.assertIn("p_value", sig["automata_vs_lstm"])


if __name__ == "__main__":
    unittest.main()
