#!/usr/bin/env python3
"""Deney JSON sonuçlarından README'deki sayısal tablo bloğunu günceller.

Bu script rapor metnini veya görselleri yazmaz; yalnızca <!-- AUTO_METRICS_START -->
ile <!-- AUTO_METRICS_END --> arasındaki tabloları outputs/results/*.json dosyalarından doldurur.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

SCENARIOS = ("original", "gaussian_noise", "unseen")


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _mean_std(values: list[float]) -> str:
    if not values:
        return "N/A"
    mean = float(np.mean(values))
    std = float(np.std(values))
    return f"{_fmt(mean)} ± {_fmt(std)}"


def _skab_models(skab_data: list[dict[str, Any]], scenario: str) -> list[str]:
    models = []
    for key in skab_data[0][scenario]:
        if key not in {"automata_states", "automata_densities"}:
            models.append(key)
    return models


def _batadal_models(batadal_data: list[dict[str, Any]], scenario: str) -> list[str]:
    models = []
    for key in batadal_data[0][scenario]:
        if key not in {"automata_states", "automata_density", "automata_densities"}:
            models.append(key)
    return models


def skab_table(skab_data: list[dict[str, Any]], scenario: str) -> str:
    models = _skab_models(skab_data, scenario)
    lines = [
        f"### SKAB — {scenario} (5 fold ort., 5 seed)",
        "",
        "| Model | Accuracy | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in models:
        acc, prec, rec, f1 = [], [], [], []
        for seed_entry in skab_data:
            mean = seed_entry[scenario][model]["mean"]
            acc.append(mean["accuracy"])
            prec.append(mean["precision"])
            rec.append(mean["recall"])
            f1.append(mean["f1"])
        lines.append(
            f"| {model.upper()} | {_mean_std(acc)} | {_mean_std(prec)} | {_mean_std(rec)} | {_mean_std(f1)} |"
        )
    lines.append("")
    return "\n".join(lines)


def batadal_table(batadal_data: list[dict[str, Any]], scenario: str) -> str:
    models = _batadal_models(batadal_data, scenario)
    lines = [
        f"### BATADAL — {scenario} (%20 test, 5 seed)",
        "",
        "| Model | Accuracy | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in models:
        acc, prec, rec, f1 = [], [], [], []
        for seed_entry in batadal_data:
            metrics = seed_entry[scenario][model]["metrics"]
            acc.append(metrics["accuracy"])
            prec.append(metrics["precision"])
            rec.append(metrics["recall"])
            f1.append(metrics["f1"])
        lines.append(
            f"| {model.upper()} | {_mean_std(acc)} | {_mean_std(prec)} | {_mean_std(rec)} | {_mean_std(f1)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _transition_density_from_matrix(transitions: dict[str, dict[str, float]]) -> float:
    n_states = len(transitions)
    if n_states <= 1:
        return 0.0
    transition_count = sum(len(targets) for targets in transitions.values())
    return transition_count / (n_states * (n_states - 1))


def _load_fold_automata_stats(
    skab_entry: dict[str, Any],
    scenario: str,
    logs_path: Path | None = None,
) -> dict[int, dict[str, float]]:
    stats: dict[int, dict[str, float]] = {}
    for output in skab_entry.get(scenario, {}).get("automata", {}).get("outputs", []):
        fold = output.get("fold")
        if fold is None:
            continue
        if "state_count" in output and "transition_density" in output:
            stats[int(fold)] = {
                "state_count": float(output["state_count"]),
                "transition_density": float(output["transition_density"]),
            }
            continue
        transitions = output.get("transition_probabilities", {})
        if transitions:
            stats[int(fold)] = {
                "state_count": float(len(transitions)),
                "transition_density": float(_transition_density_from_matrix(transitions)),
            }

    if stats or logs_path is None or not logs_path.exists():
        return stats

    seed = skab_entry.get("seed")
    for line in logs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if (
            record.get("dataset") != "skab"
            or record.get("model") != "automata"
            or record.get("scenario") != scenario
            or record.get("seed") != seed
        ):
            continue
        params = record.get("parameters", {})
        fold = params.get("fold")
        if fold is None:
            continue
        stats[int(fold)] = {
            "state_count": float(params.get("state_count", 0)),
            "transition_density": float(params.get("transition_density", 0.0)),
        }
    return stats


def skab_fold_comparison_table(
    skab_data: list[dict[str, Any]],
    *,
    scenario: str = "original",
    seed: int = 42,
) -> str:
    entry = next(item for item in skab_data if item["seed"] == seed)
    models = _skab_models(skab_data, scenario)
    n_folds = len(entry[scenario][models[0]]["raw"])

    header = "| Fold | " + " | ".join(model.upper() + " F1" for model in models) + " |"
    sep = "|---:|" + "|".join(["---:"] * len(models)) + "|"
    lines = [
        f"### SKAB fold sonuçları (seed={seed}, {scenario})",
        "",
        header,
        sep,
    ]
    for fold_idx in range(n_folds):
        values = [_fmt(entry[scenario][model]["raw"][fold_idx]["f1"]) for model in models]
        lines.append(f"| {fold_idx} | " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def skab_fold_detailed_tables(
    skab_data: list[dict[str, Any]],
    logs_path: Path,
    *,
    scenario: str = "original",
    seed: int = 42,
) -> str:
    entry = next(item for item in skab_data if item["seed"] == seed)
    models = _skab_models(skab_data, scenario)
    automata_stats = _load_fold_automata_stats(entry, scenario, logs_path)
    sections: list[str] = [
        f"### SKAB fold detayı (seed={seed}, {scenario})",
        "",
    ]

    for model in models:
        raw = entry[scenario][model]["raw"]
        lines = [
            f"#### {model.upper()}",
            "",
            "| Fold | Acc | Prec | Rec | F1 |",
            "|---:|---:|---:|---:|---:|",
        ]
        for fold_idx, fold_metrics in enumerate(raw):
            lines.append(
                f"| {fold_idx} | {_fmt(fold_metrics['accuracy'])} | {_fmt(fold_metrics['precision'])} | "
                f"{_fmt(fold_metrics['recall'])} | {_fmt(fold_metrics['f1'])} |"
            )
        if model == "automata":
            lines.extend(["", "| Fold | State | Density |", "|---:|---:|---:|"])
            for fold_idx in range(len(raw)):
                extra = automata_stats.get(fold_idx, {})
                state_count = int(extra["state_count"]) if "state_count" in extra else "—"
                density = _fmt(extra["transition_density"]) if "transition_density" in extra else "—"
                lines.append(f"| {fold_idx} | {state_count} | {density} |")
        lines.append("")
        sections.append("\n".join(lines))

    return "\n".join(sections)


def parameter_variation_table(variation_path: Path, dataset_label: str) -> str:
    if not variation_path.exists():
        return ""
    data = json.loads(variation_path.read_text(encoding="utf-8"))
    if not data:
        return ""

    lines = [
        f"### {dataset_label} parametre taraması",
        "",
        "| W | α | F1 | State | Density |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(data, key=lambda item: (item["window_size"], item["alphabet_size"])):
        lines.append(
            f"| {row['window_size']} | {row['alphabet_size']} | "
            f"{_fmt(row['f1_mean'])} ± {_fmt(row['f1_std'])} | "
            f"{_fmt(row['states_mean'])} | {_fmt(row['density_mean'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def significance_table(significance: dict[str, Any]) -> str:
    lines = [
        "### İstatistiksel testler (SKAB, original)",
        "",
        "| Karşılaştırma | Test | İstatistik | p |",
        "|---|---|---|---:|",
    ]
    if "automata_vs_lstm" in significance:
        row = significance["automata_vs_lstm"]
        lines.append(
            f"| Otomata vs LSTM | Wilcoxon (F1) | {row['statistic']:.4f} | {row['p_value']:.4f} |"
        )
    if "automata_vs_gru" in significance:
        row = significance["automata_vs_gru"]
        lines.append(
            f"| Otomata vs GRU | Wilcoxon (F1) | {row['statistic']:.4f} | {row['p_value']:.4f} |"
        )
    if "mcnemar_automata_vs_lstm" in significance:
        row = significance["mcnemar_automata_vs_lstm"]
        lines.append(
            f"| Otomata vs LSTM | McNemar (b={row['b']}, c={row['c']}) | — | {row['p_value']:.4f} |"
        )
    if "mcnemar_automata_vs_gru" in significance:
        row = significance["mcnemar_automata_vs_gru"]
        lines.append(
            f"| Otomata vs GRU | McNemar (b={row['b']}, c={row['c']}) | — | {row['p_value']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_metrics_block(
    skab_data: list[dict[str, Any]],
    batadal_data: list[dict[str, Any]],
    significance: dict[str, Any],
    logs_path: Path,
    results_dir: Path,
) -> str:
    sections = [
        "<!-- AUTO_METRICS_START -->",
        "## Ek: Sayısal sonuçlar",
        "",
        "Özet tablolar `outputs/results/*.json` dosyalarından üretilir. Deneyleri yeniden koştuktan sonra",
        "`python3 scripts/generate_report_metrics.py` ile bu bölümü güncelleyebilirsiniz.",
        "",
    ]
    for scenario in SCENARIOS:
        sections.append(skab_table(skab_data, scenario))
        sections.append(batadal_table(batadal_data, scenario))

    sections.append(skab_fold_comparison_table(skab_data, scenario="original", seed=42))
    sections.append(skab_fold_detailed_tables(skab_data, logs_path, scenario="original", seed=42))
    sections.append(parameter_variation_table(results_dir / "automata_parameter_variation.json", "SKAB"))
    sections.append(parameter_variation_table(results_dir / "batadal_parameter_variation.json", "BATADAL"))
    sections.append(significance_table(significance))
    sections.append("<!-- AUTO_METRICS_END -->")
    return "\n".join(sections)


def replace_block(readme: str, start: str, end: str, new_block: str) -> str:
    if start in readme and end in readme:
        before = readme.split(start)[0]
        after = readme.split(end)[1]
        return before + new_block + after
    return readme.rstrip() + "\n\n" + new_block + "\n"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "outputs" / "results"
    skab_path = results_dir / "skab_experiments.json"
    batadal_path = results_dir / "batadal_experiments.json"
    sig_path = results_dir / "statistical_significance.json"
    logs_path = root / "outputs" / "logs" / "experiments.jsonl"

    if not skab_path.exists() or not batadal_path.exists():
        raise SystemExit("Deney sonuçları yok. Önce `python3 run_experiments.py` çalıştırın.")

    skab_data = json.loads(skab_path.read_text(encoding="utf-8"))
    batadal_data = json.loads(batadal_path.read_text(encoding="utf-8"))
    significance = json.loads(sig_path.read_text(encoding="utf-8")) if sig_path.exists() else {}

    readme_path = root / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    readme = replace_block(
        readme,
        "<!-- AUTO_METRICS_START -->",
        "<!-- AUTO_METRICS_END -->",
        build_metrics_block(skab_data, batadal_data, significance, logs_path, results_dir),
    )

    # Eski otomatik görsel bloğu varsa temizle (artık üretilmiyor)
    if "<!-- AUTO_FIGURES_START -->" in readme:
        before = readme.split("<!-- AUTO_FIGURES_START -->")[0]
        after = readme.split("<!-- AUTO_FIGURES_END -->")[1] if "<!-- AUTO_FIGURES_END -->" in readme else ""
        readme = before.rstrip() + "\n\n---\n" + after.lstrip("\n")

    readme_path.write_text(readme, encoding="utf-8")
    print(f"README tabloları güncellendi: {readme_path}")


if __name__ == "__main__":
    main()
