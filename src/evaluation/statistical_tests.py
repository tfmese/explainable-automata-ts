from __future__ import annotations

import logging
from typing import Any

from scipy.stats import binomtest, wilcoxon

logger = logging.getLogger(__name__)


def calculate_statistical_significance(skab_results_by_seed: list[dict[str, Any]]) -> dict[str, Any]:
    logger.info("Computing Wilcoxon and McNemar tests...")
    automata_vs_lstm_a: list[float] = []
    automata_vs_lstm_b: list[float] = []
    automata_vs_gru_a: list[float] = []
    automata_vs_gru_b: list[float] = []

    mcnemar_automata_vs_lstm = {"b": 0, "c": 0}
    mcnemar_automata_vs_gru = {"b": 0, "c": 0}

    def _pool_mcnemar_counts(
        orig: dict[str, Any], model_a: str, model_b: str, acc: dict[str, int]
    ) -> None:
        if model_a not in orig or model_b not in orig:
            return
        a_outputs = orig[model_a].get("outputs", [])
        b_outputs = orig[model_b].get("outputs", [])
        if not a_outputs or not b_outputs:
            return

        a_by_fold = {o.get("fold"): o for o in a_outputs if "fold" in o}
        b_by_fold = {o.get("fold"): o for o in b_outputs if "fold" in o}
        common_folds = sorted(set(a_by_fold).intersection(b_by_fold))
        for fold_id in common_folds:
            a_preds = a_by_fold[fold_id].get("predictions", [])
            b_preds = b_by_fold[fold_id].get("predictions", [])
            if not a_preds or not b_preds:
                continue

            a_map = {r["time_step"]: (r["y_true"], r["y_pred"]) for r in a_preds if "time_step" in r}
            b_map = {r["time_step"]: (r["y_true"], r["y_pred"]) for r in b_preds if "time_step" in r}
            common_ts = sorted(set(a_map).intersection(b_map))
            if not common_ts:
                continue

            for ts in common_ts:
                y_true_a, pred_a = a_map[ts]
                y_true_b, pred_b = b_map[ts]
                if y_true_a != y_true_b:
                    continue

                if pred_a == 1 and pred_b == 0:
                    acc["b"] += 1
                elif pred_a == 0 and pred_b == 1:
                    acc["c"] += 1

    def _paired_fold_f1(orig: dict[str, Any], model_a: str, model_b: str) -> tuple[list[float], list[float]]:
        a_raw = orig.get(model_a, {}).get("raw", [])
        b_raw = orig.get(model_b, {}).get("raw", [])
        if len(a_raw) != len(b_raw) or not a_raw:
            return [], []
        return [float(item["f1"]) for item in a_raw], [float(item["f1"]) for item in b_raw]

    for seed_res in skab_results_by_seed:
        orig = seed_res.get("original", {})

        a_lstm, l_lstm = _paired_fold_f1(orig, "automata", "lstm")
        automata_vs_lstm_a.extend(a_lstm)
        automata_vs_lstm_b.extend(l_lstm)

        a_gru, l_gru = _paired_fold_f1(orig, "automata", "gru")
        automata_vs_gru_a.extend(a_gru)
        automata_vs_gru_b.extend(l_gru)

        _pool_mcnemar_counts(orig, "automata", "lstm", mcnemar_automata_vs_lstm)
        _pool_mcnemar_counts(orig, "automata", "gru", mcnemar_automata_vs_gru)

    sig_results: dict[str, Any] = {
        "paired_fold_count_lstm": len(automata_vs_lstm_a),
        "paired_fold_count_gru": len(automata_vs_gru_a),
    }
    try:
        if len(automata_vs_lstm_a) >= 2 and len(automata_vs_lstm_b) == len(automata_vs_lstm_a):
            stat_al, p_val_al = wilcoxon(automata_vs_lstm_a, automata_vs_lstm_b)
            sig_results["automata_vs_lstm"] = {
                "statistic": float(stat_al),
                "p_value": float(p_val_al),
                "n_pairs": len(automata_vs_lstm_a),
            }

        if len(automata_vs_gru_a) >= 2 and len(automata_vs_gru_b) == len(automata_vs_gru_a):
            stat_ag, p_val_ag = wilcoxon(automata_vs_gru_a, automata_vs_gru_b)
            sig_results["automata_vs_gru"] = {
                "statistic": float(stat_ag),
                "p_value": float(p_val_ag),
                "n_pairs": len(automata_vs_gru_a),
            }

        n01_lstm = int(mcnemar_automata_vs_lstm["b"])
        n10_lstm = int(mcnemar_automata_vs_lstm["c"])
        n_lstm = n01_lstm + n10_lstm
        if n_lstm > 0:
            res_lstm = binomtest(k=n01_lstm, n=n_lstm, p=0.5, alternative="two-sided")
            sig_results["mcnemar_automata_vs_lstm"] = {
                "b": n01_lstm,
                "c": n10_lstm,
                "n": n_lstm,
                "p_value": float(res_lstm.pvalue),
            }

        n01_gru = int(mcnemar_automata_vs_gru["b"])
        n10_gru = int(mcnemar_automata_vs_gru["c"])
        n_gru = n01_gru + n10_gru
        if n_gru > 0:
            res_gru = binomtest(k=n01_gru, n=n_gru, p=0.5, alternative="two-sided")
            sig_results["mcnemar_automata_vs_gru"] = {
                "b": n01_gru,
                "c": n10_gru,
                "n": n_gru,
                "p_value": float(res_gru.pvalue),
            }
    except Exception as exc:
        logger.warning("Could not compute significance tests: %s", exc)
        sig_results["error"] = str(exc)

    return sig_results
