import unittest
import numpy as np
from src.config import load_config
from src.models.deep_learning_pipeline import TimeSeriesDataset, DeepLearningPipeline, set_seed

class TestDeepLearningPipeline(unittest.TestCase):
    def setUp(self):
        set_seed(42)
        self.X = np.random.randn(100, 3)
        self.y = np.random.randint(0, 2, size=100)
        self.window_size = 4
        self.config = load_config("config/config.yaml")

    def test_dataset_windowing(self):
        dataset = TimeSeriesDataset(self.X, self.y, self.window_size)
        self.assertEqual(len(dataset), 100 - self.window_size + 1)
        
        x_win, y_target = dataset[0]
        self.assertEqual(x_win.shape, (self.window_size, 3))

        np.testing.assert_allclose(x_win.numpy(), self.X[:self.window_size], rtol=1e-5, atol=1e-5)
        self.assertEqual(y_target.item(), self.y[self.window_size - 1])

    def test_lstm_pipeline_training_and_prediction(self):
        pipeline = DeepLearningPipeline(
            config=self.config,
            model_name="lstm",
            input_size=3,
            window_size=self.window_size
        )
        
        pipeline.fit(self.X[:50], self.y[:50], val_X=self.X[50:70], val_y=self.y[50:70])
        
        preds = pipeline.predict(self.X[70:])

        self.assertEqual(len(preds), len(self.X[70:]) - self.window_size + 1)
        
        metrics = pipeline.predict_and_evaluate(self.X[70:], self.y[70:])
        self.assertIn("accuracy", metrics)
        self.assertIn("f1", metrics)

    def test_cnn_pipeline_training_and_prediction(self):
        pipeline = DeepLearningPipeline(
            config=self.config,
            model_name="cnn1d",
            input_size=3,
            window_size=self.window_size
        )
        
        # Eğitme
        pipeline.fit(self.X[:40], self.y[:40])
        
        # Öngörme
        preds = pipeline.predict(self.X[40:])
        self.assertEqual(len(preds), len(self.X[40:]) - self.window_size + 1)

if __name__ == "__main__":
    unittest.main()
