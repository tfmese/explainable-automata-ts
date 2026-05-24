import unittest

from src.explainability.automata_explainer import explain_automata_decision
from src.models.automata import ProbabilisticAutomata


class TestUnseenMapping(unittest.TestCase):
    def test_unseen_pattern_maps_to_nearest_seen_pattern(self):
        automata = ProbabilisticAutomata().fit(["aaa", "abc", "bcc", "abc"])

        mapping = automata.map_pattern("adc")

        self.assertEqual(mapping.status, "unseen")
        self.assertEqual(mapping.mapped, "abc")
        self.assertEqual(mapping.distance, 1)

    def test_explanation_reports_unseen_mapping(self):
        automata = ProbabilisticAutomata().fit(["aab", "abc", "bcc", "abc"])

        explanation = explain_automata_decision(automata, ["aab", "adc"], time_step=5)

        self.assertEqual(explanation["time_step"], 5)
        self.assertEqual(explanation["status"], "unseen")
        self.assertEqual(explanation["mapped_to"], "abc")
        self.assertIn("transitions", explanation)
        self.assertIn("confidence", explanation)


class TestProbabilisticAutomataMetrics(unittest.TestCase):
    def test_state_count_matches_unique_patterns(self):
        patterns = ["aaa", "aab", "abb", "aab"]
        automata = ProbabilisticAutomata().fit(patterns)

        self.assertEqual(automata.state_count, len(set(patterns)))

    def test_transition_density_is_between_zero_and_one(self):
        automata = ProbabilisticAutomata().fit(["aaa", "aab", "abb", "aab"])

        self.assertGreaterEqual(automata.transition_density, 0.0)
        self.assertLessEqual(automata.transition_density, 1.0)


if __name__ == "__main__":
    unittest.main()
