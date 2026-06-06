import unittest

from src.explainability.explanation_export import select_explanation_samples


class TestExplanationExport(unittest.TestCase):
    def test_unseen_scenario_prioritizes_unseen_samples(self):
        explanations = [
            {"time_step": i, "status": "seen", "pattern": f"p{i}"}
            for i in range(100)
        ]
        explanations[50] = {
            "time_step": 50,
            "status": "unseen",
            "pattern": "adc",
            "mapped_to": "abc",
        }

        selected = select_explanation_samples(explanations, "unseen", max_count=5)

        self.assertTrue(any(item.get("status") == "unseen" for item in selected))
        self.assertLessEqual(len(selected), 5)

    def test_original_scenario_keeps_prefix(self):
        explanations = [{"time_step": i, "status": "seen"} for i in range(10)]
        selected = select_explanation_samples(explanations, "original", max_count=3)
        self.assertEqual([0, 1, 2], [item["time_step"] for item in selected])


if __name__ == "__main__":
    unittest.main()
