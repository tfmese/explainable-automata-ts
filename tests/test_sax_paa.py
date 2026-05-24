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


class TestPaaValidation(unittest.TestCase):
    def test_paa_rejects_non_positive_segments(self):
        sequence = np.array([1.0, 2.0, 3.0, 4.0])

        with self.assertRaises(ValueError):
            paa(sequence, segments=0)

        with self.assertRaises(ValueError):
            paa(sequence, segments=-1)

    def test_paa_rejects_segments_larger_than_sequence(self):
        with self.assertRaises(ValueError):
            paa(np.array([1.0, 2.0, 3.0, 4.0]), segments=5)


class TestSaxValidation(unittest.TestCase):
    def test_sax_rejects_alphabet_size_below_two(self):
        sequence = np.array([-2.0, 0.0, 2.0])

        with self.assertRaises(ValueError):
            sax(sequence, alphabet_size=1)

        with self.assertRaises(ValueError):
            sax(sequence, alphabet_size=0)

    def test_sax_rejects_alphabet_size_above_twenty_six(self):
        with self.assertRaises(ValueError):
            sax(np.array([-2.0, 0.0, 2.0]), alphabet_size=27)


if __name__ == "__main__":
    unittest.main()
