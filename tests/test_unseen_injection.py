import unittest

from src.experiments.scenarios import inject_unseen_pattern


class TestUnseenInjection(unittest.TestCase):
    def test_injected_pattern_not_in_forbidden_vocabulary(self):
        # alphabet_size=3 => {a,b,c}, length=4 => 3^4 küçük; deterministik seçim yapılır.
        forbidden = {"aaaa", "aaab"}
        patterns = ["aaaa", "aaab", "aabb", "abcc", "bbbb"]

        mutated = inject_unseen_pattern(
            patterns,
            replacement="zzzz",
            forbidden_patterns=forbidden,
            alphabet_size=3,
        )

        injected = mutated[len(patterns) // 2]
        self.assertNotIn(injected, forbidden)
        self.assertEqual(len(injected), len(patterns[0]))


if __name__ == "__main__":
    unittest.main()

