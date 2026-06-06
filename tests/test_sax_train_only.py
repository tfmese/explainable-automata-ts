import unittest

import numpy as np

from src.features.sax import TrainableSAXEncoder


class TestTrainableSAXEncoder(unittest.TestCase):
    def test_encoder_fit_uses_only_training_statistics(self):
        train = np.linspace(0.0, 10.0, 200)
        encoder = TrainableSAXEncoder().fit(
            train,
            window_size=5,
            paa_segments=5,
            alphabet_size=3,
        )

        train_mean = encoder.mean_
        train_std = encoder.std_

        shifted_test = np.linspace(100.0, 200.0, 100)
        patterns = encoder.transform(shifted_test)

        self.assertGreater(len(patterns), 0)
        self.assertAlmostEqual(encoder.mean_, train_mean)
        self.assertAlmostEqual(encoder.std_, train_std)

    def test_unfitted_encoder_raises_on_transform(self):
        encoder = TrainableSAXEncoder()
        with self.assertRaises(RuntimeError):
            encoder.transform(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))


if __name__ == "__main__":
    unittest.main()
