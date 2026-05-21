import unittest

import numpy as np
import pandas as pd

from src.data.preprocessing import LeakageSafePreprocessor
from src.data.splits import (
    assert_no_group_leakage,
    make_batadal_chronological_split,
    make_skab_group_folds,
)


class TestSplitsNoLeakage(unittest.TestCase):
    def test_skab_group_folds_keep_source_files_disjoint(self):
        df = pd.DataFrame(
            {
                "value": np.arange(12),
                "anomaly": [0, 1] * 6,
                "source_file": ["f1"] * 3 + ["f2"] * 3 + ["f3"] * 3 + ["f4"] * 3,
            }
        )

        folds = make_skab_group_folds(
            df,
            target_column="anomaly",
            group_column="source_file",
            n_splits=2,
            seed=42,
            prefer_stratified=False,
        )

        for split in folds:
            assert_no_group_leakage(df, split, "source_file")

    def test_batadal_split_is_chronological_60_20_20(self):
        df = pd.DataFrame({"x": np.arange(10)})

        split = make_batadal_chronological_split(df)

        np.testing.assert_array_equal(split.train, np.arange(0, 6))
        np.testing.assert_array_equal(split.validation, np.arange(6, 8))
        np.testing.assert_array_equal(split.test, np.arange(8, 10))

    def test_preprocessor_fits_only_train_statistics(self):
        train = pd.DataFrame({"x": [0.0, 2.0], "y": [10.0, 14.0]})
        test = pd.DataFrame({"x": [100.0], "y": [200.0]})
        preprocessor = LeakageSafePreprocessor(use_standard_scaler=True, pca_components=None)

        preprocessor.fit(train, ["x", "y"])
        transformed_test = preprocessor.transform(test)

        self.assertAlmostEqual(preprocessor.scaler_.mean_[0], 1.0)
        self.assertGreater(transformed_test[0, 0], 50.0)


if __name__ == "__main__":
    unittest.main()
