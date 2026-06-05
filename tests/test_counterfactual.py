import unittest

from src.explainability.automata_explainer import explain_automata_decision, find_counterfactual
from src.models.automata import ProbabilisticAutomata


class TestCounterfactualAnalysis(unittest.TestCase):
    def test_counterfactual_finds_decision_flipping_pattern(self):
        # Durumlar: 'aab' -> 'abc' (1.0 geçişi)
        # 'aab' -> 'bcc' (0.0 geçişi - imkansız)
        automata = ProbabilisticAutomata(anomaly_quantile=0.5).fit(["aab", "abc", "aab", "abc"])
        automata.threshold_ = 0.1

        # Test: 'abc' -> 'abc' dizisi
        # 'abc' -> 'abc' geçişi train'de olmadığı için olasılık çok düşük (1e-12) olacaktır ve ANOMALİ kararı verilecektir.
        patterns = ["abc", "abc"]
        explanation = explain_automata_decision(automata, patterns, time_step=1)

        self.assertEqual(explanation["decision"], "anomaly")
        cf = explanation["counterfactual"]
        
        # Karar 'anomaly' olduğu için counterfactual bunu 'normal' yapmalıdır.
        self.assertIsNotNone(cf)
        self.assertEqual(cf["decision"], "normal")
        self.assertEqual(cf["pattern"], "aab")  # 'abc' -> 'aab' geçişi normaldir.
        self.assertEqual(cf["distance"], 2)  # abc ve aab arasındaki Levenshtein mesafesi 2'dir.

    def test_counterfactual_returns_none_if_no_states(self):
        automata = ProbabilisticAutomata()
        cf = find_counterfactual(automata, ["abc"], "normal", 0.05)
        self.assertIsNone(cf)


if __name__ == "__main__":
    unittest.main()
