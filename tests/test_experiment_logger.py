import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.experiment_logger import ExperimentLogger, build_experiment_record


class TestExperimentLogger(unittest.TestCase):
    def test_logs_jsonl_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = ExperimentLogger(tmp)
            logger.log(
                build_experiment_record(
                    dataset="skab",
                    model="automata",
                    scenario="original",
                    seed=42,
                    parameters={"window_size": 4},
                    metrics={"f1": 0.5},
                    fold=0,
                )
            )
            logger.save_run_summary({"seeds": [42]})

            lines = logger.jsonl_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["dataset"], "skab")
            self.assertEqual(record["metrics"]["f1"], 0.5)

            summary = json.loads((Path(tmp) / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["seeds"], [42])


if __name__ == "__main__":
    unittest.main()
