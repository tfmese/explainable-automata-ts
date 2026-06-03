import unittest

from src.experiments.results import ExperimentResult


class TestExperimentResult(unittest.TestCase):
    def test_experiment_result_can_be_created_and_read_metrics(self):
        result = ExperimentResult(
            dataset="skab",
            model="automata",
            scenario="original",
            seed=42,
            parameters={"window_size": 4, "alphabet_size": 3},
            metrics={"f1": 0.91, "accuracy": 0.95},
        )

        self.assertEqual(result.dataset, "skab")
        self.assertEqual(result.metrics["f1"], 0.91)
        self.assertEqual(result.to_dict()["model"], "automata")


if __name__ == "__main__":
    unittest.main()
