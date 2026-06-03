import unittest

import numpy as np
import pandas as pd

from src.data.preprocessing import LeakageSafePreprocessor, build_leakage_safe_preprocessor
from src.config import load_config


class TestPreprocessingMissing(unittest.TestCase):
    def test_median_imputation_uses_train_statistics_only(self):
        train = pd.DataFrame({"sensor_a": [1.0, np.nan, 3.0], "sensor_b": [4.0, 5.0, 6.0]})
        test = pd.DataFrame({"sensor_a": [np.nan, 10.0], "sensor_b": [7.0, 8.0]})

        preprocessor = LeakageSafePreprocessor(
            use_standard_scaler=False,
            pca_components=None,
            missing_strategy="median",
        )
        preprocessor.fit(train, ["sensor_a", "sensor_b"])
        transformed = preprocessor.transform(test)

        self.assertEqual(transformed.shape, (2, 2))
        self.assertFalse(np.isnan(transformed).any())
        self.assertEqual(preprocessor._fill_values_["sensor_a"], 2.0)

    def test_build_preprocessor_reads_config(self):
        config = load_config()
        dl_preprocessor = build_leakage_safe_preprocessor(config)
        automata_preprocessor = build_leakage_safe_preprocessor(config, for_automata=True)
        self.assertEqual(dl_preprocessor.missing_strategy, "median")
        self.assertIsNone(dl_preprocessor.pca_components)
        self.assertEqual(automata_preprocessor.pca_components, 1)

    def test_dl_preprocessor_keeps_feature_dimension(self):
        config = load_config()
        train = pd.DataFrame(
            {
                "sensor_a": [0.0, 1.0, 2.0, 3.0],
                "sensor_b": [1.0, 2.0, 3.0, 4.0],
                "sensor_c": [2.0, 3.0, 4.0, 5.0],
            }
        )
        preprocessor = build_leakage_safe_preprocessor(config)
        out = preprocessor.fit_transform(train, ["sensor_a", "sensor_b", "sensor_c"])
        self.assertEqual(out.shape[1], 3)


if __name__ == "__main__":
    unittest.main()
