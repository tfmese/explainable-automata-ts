import unittest

import pandas as pd

from src.config import load_config
from src.pipeline import build_fixed_automata_pipeline


class TestConfigAndPipeline(unittest.TestCase):
    def test_config_loads_without_pyyaml(self):
        config = load_config()

        self.assertEqual(config.get("project", "name"), "explainable-automata-ts")
        self.assertEqual(config.get("deep_learning", "batch_size"), 32)

    def test_fixed_automata_pipeline_runs_with_train_only_fit(self):
        config = load_config()
        train = pd.DataFrame({"sensor_a": [0, 1, 2, 3, 4, 5], "sensor_b": [1, 2, 3, 4, 5, 6]})
        test = pd.DataFrame({"sensor_a": [2, 3, 4, 5, 6], "sensor_b": [3, 4, 5, 6, 7]})

        pipeline = build_fixed_automata_pipeline(config).fit(train, ["sensor_a", "sensor_b"])
        prediction = pipeline.predict(test)

        self.assertIn(prediction["decision"], {"normal", "anomaly"})
        self.assertIn("probability", prediction)


if __name__ == "__main__":
    unittest.main()
