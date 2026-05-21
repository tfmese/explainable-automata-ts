import unittest

import numpy as np

from src.features.paa import paa
from src.features.sax import sax
from src.features.windowing import extract_sax_patterns


class TestSaxPaa(unittest.TestCase):
    def test_paa_reduces_to_requested_segments(self):
        reduced = paa(np.array([1, 2, 3, 4]), segments=2)

        np.testing.assert_allclose(reduced, np.array([1.5, 3.5]))

    def test_sax_uses_configured_alphabet_size(self):
        pattern = sax(np.array([-2.0, 0.0, 2.0]), alphabet_size=3)

        self.assertEqual(len(pattern), 3)
        self.assertTrue(set(pattern).issubset({"a", "b", "c"}))

    def test_sliding_pattern_count(self):
        patterns = extract_sax_patterns(
            np.array([1, 2, 3, 4, 5]),
            window_size=3,
            paa_segments=3,
            alphabet_size=3,
        )

        self.assertEqual(len(patterns), 3)


if __name__ == "__main__":
    unittest.main()
