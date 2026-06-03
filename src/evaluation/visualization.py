from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve

logger = logging.getLogger("visualization")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Outfit", "Inter", "Roboto", "Helvetica"],
        "figure.facecolor": "#0B0F19",
        "axes.facecolor": "#111827",
        "axes.edgecolor": "#374151",
        "axes.labelcolor": "#E5E7EB",
        "xtick.color": "#9CA3AF",
        "ytick.color": "#9CA3AF",
        "text.color": "#F9FAFB",
        "grid.color": "#1F2937",
        "grid.linestyle": "--",
    }
)


def plot_confusion_matrix(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    title: str,
    save_path: Path,
) -> None:
    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="mako",
        cbar=False,
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor="#0B0F19")
    plt.close()


def plot_transition_heatmap(
    transition_probabilities: dict[str, dict[str, float]],
    save_path: Path,
) -> None:
    # Durum geçiş olasılıklarını temsil eden bir kare matris 
    states = sorted(list(transition_probabilities.keys()))
    if not states:
        return

    matrix = np.zeros((len(states), len(states)))
    state_to_idx = {s: i for i, s in enumerate(states)}

    for src, targets in transition_probabilities.items():
        src_idx = state_to_idx[src]
        for tgt, prob in targets.items():
            if tgt in state_to_idx:
                matrix[src_idx, state_to_idx[tgt]] = prob

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        xticklabels=states,
        yticklabels=states,
        cmap="rocket_r",
        annot=len(states) <= 12,
        fmt=".2f",
    )
    plt.title("Automata State Transition Probability Heatmap", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Target State", fontsize=12)
    plt.ylabel("Source State", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor="#0B0F19")
    plt.close()


def plot_automata_state_diagram(
    transition_probabilities: dict[str, dict[str, float]],
    save_path: Path,
    max_states: int = 24,
) -> None:
    states = sorted(transition_probabilities.keys())[:max_states]
    if not states:
        return

    angles = np.linspace(0, 2 * np.pi, len(states), endpoint=False)
    positions = {state: np.array([np.cos(angle), np.sin(angle)]) for state, angle in zip(states, angles)}

    plt.figure(figsize=(9, 9))
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.axis("off")

    for source in states:
        start = positions[source]
        for target, probability in transition_probabilities.get(source, {}).items():
            if target not in positions:
                continue
            end = positions[target]
            delta = end - start
            if np.linalg.norm(delta) == 0:
                continue
            ax.annotate(
                "",
                xy=end * 0.88,
                xytext=start * 0.88,
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#60A5FA",
                    "alpha": max(0.25, min(0.95, probability)),
                    "lw": 0.8 + 2.0 * probability,
                },
            )

    for state, pos in positions.items():
        circle = plt.Circle(pos, 0.09, color="#F9FAFB", ec="#22C55E", lw=2, zorder=3)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], state, ha="center", va="center", color="#111827", fontsize=8, zorder=4)

    plt.title("Automata State Diagram", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor="#0B0F19")
    plt.close()


def plot_parameter_sensitivity(
    variation_data: list[dict[str, Any]],
    save_path: Path,
) -> None:
    # Pencere boyutu ve alfabe boyutunun F1 skoru üzerindeki duyarlılık grafiklerini çixiyoruz
    df = pd.DataFrame(variation_data)
    if df.empty:
        return

    plt.figure(figsize=(12, 5))

    # 1.Alt Grafik: Pencere boyutu vs F1
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df, x="window_size", y="f1_mean", hue="alphabet_size", marker="o", palette="viridis")
    plt.title("Effect of Window Size on F1 Score", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Window Size", fontsize=10)
    plt.ylabel("Mean F1 Score", fontsize=10)

    # 2.Alt Grafik: Alfabe boyutu vs F1
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df, x="alphabet_size", y="f1_mean", hue="window_size", marker="s", palette="magma")
    plt.title("Effect of Alphabet Size on F1 Score", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Alphabet Size", fontsize=10)
    plt.ylabel("Mean F1 Score", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor="#0B0F19")
    plt.close()


def plot_state_sensitivity(
    variation_data: list[dict[str, Any]],
    save_path: Path,
) -> None:
    # Parametrelerin state sayısı ve geçiş yoğunluğu üzerindeki etkilerini görselleştiriyoruz
    df = pd.DataFrame(variation_data)
    if df.empty:
        return

    plt.figure(figsize=(12, 5))

    # 1.Alt Grafik: Parametrelerin Durum Sayısına Etkisi
    plt.subplot(1, 2, 1)
    sns.barplot(data=df, x="window_size", y="states_mean", hue="alphabet_size", palette="mako")
    plt.title("State Count Sensitivity", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Window Size", fontsize=10)
    plt.ylabel("Automata State Count", fontsize=10)

    # 2. Alt Grafik: Parametrelerin Geçiş Yoğunluğuna Etkisi
    plt.subplot(1, 2, 2)
    sns.barplot(data=df, x="window_size", y="density_mean", hue="alphabet_size", palette="flare")
    plt.title("Transition Density Sensitivity", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Window Size", fontsize=10)
    plt.ylabel("Transition Density", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor="#0B0F19")
    plt.close()


def plot_precision_recall_from_records(records: list[dict[str, Any]], title: str, save_path: Path) -> None:
    if not records:
        return
    y_true = np.asarray([row["y_true"] for row in records])
    scores = np.asarray([row["anomaly_score"] for row in records])
    if len(np.unique(y_true)) < 2:
        return

    precision, recall, _ = precision_recall_curve(y_true, scores)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="#22C55E", lw=2)
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor="#0B0F19")
    plt.close()


def plot_roc_from_records(records: list[dict[str, Any]], title: str, save_path: Path) -> None:
    if not records:
        return

    y_true = np.asarray([row["y_true"] for row in records])
    scores = np.asarray([row["anomaly_score"] for row in records])

    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = float(auc(fpr, tpr))

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#3B82F6", lw=2, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="#9CA3AF", lw=1, linestyle="--")
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.legend(frameon=True, facecolor="#111827", edgecolor="#374151")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor="#0B0F19")
    plt.close()


def emit_model_diagnostics(
    dataset_name: str,
    scenario_data: dict[str, Any],
    figures_path: Path,
    scenario_suffix: str | None = None,
) -> None:
    suffix = f"_{scenario_suffix}" if scenario_suffix else ""
    for model_name, model_data in scenario_data.items():
        if model_name in {"automata_states", "automata_densities", "automata_density"}:
            continue

        if "mean" in model_data:
            raw_metrics = model_data.get("raw", [])
            metrics = (
                {
                    "tp": sum(item.get("tp", 0) for item in raw_metrics),
                    "fp": sum(item.get("fp", 0) for item in raw_metrics),
                    "fn": sum(item.get("fn", 0) for item in raw_metrics),
                    "tn": sum(item.get("tn", 0) for item in raw_metrics),
                }
                if raw_metrics
                else model_data["mean"]
            )
            outputs = model_data.get("outputs", [])
            first_output = outputs[0] if outputs else {}
        else:
            metrics = model_data.get("metrics", {})
            first_output = model_data

        required = {"tp", "fp", "fn", "tn"}
        if required.issubset(metrics):
            plot_confusion_matrix(
                int(metrics["tp"]),
                int(metrics["fp"]),
                int(metrics["fn"]),
                int(metrics["tn"]),
                f"{dataset_name.upper()} {model_name.upper()} Confusion Matrix",
                figures_path / f"{dataset_name}_{model_name}{suffix}_confusion_matrix.png",
            )

        records = first_output.get("predictions", [])
        plot_precision_recall_from_records(
            records,
            f"{dataset_name.upper()} {model_name.upper()} Precision-Recall",
            figures_path / f"{dataset_name}_{model_name}{suffix}_precision_recall.png",
        )
        plot_roc_from_records(
            records,
            f"{dataset_name.upper()} {model_name.upper()} ROC Curve",
            figures_path / f"{dataset_name}_{model_name}{suffix}_roc_curve.png",
        )

        if model_name == "automata":
            transitions = first_output.get("transition_probabilities", {})
            plot_transition_heatmap(
                transitions,
                figures_path / f"{dataset_name}_automata{suffix}_transition_heatmap.png",
            )
            plot_automata_state_diagram(
                transitions,
                figures_path / f"{dataset_name}_automata{suffix}_state_diagram.png",
            )


def generate_all_visualizations(results_root: str | Path, figures_root: str | Path) -> None:
    results_path = Path(results_root)
    figures_path = Path(figures_root)
    figures_path.mkdir(parents=True, exist_ok=True)

    # 1. Aşama: Parametre Duyarlılık Çizimleri (Otomata modeli için grid arama sonuçları)
    skab_var_file = results_path / "automata_parameter_variation.json"
    if skab_var_file.exists():
        logger.info("Generating SKAB parameter sensitivity plots...")
        with open(skab_var_file) as f:
            variation_data = json.load(f)
        plot_parameter_sensitivity(variation_data, figures_path / "parameter_sensitivity.png")
        plot_state_sensitivity(variation_data, figures_path / "state_sensitivity.png")

    batadal_var_file = results_path / "batadal_parameter_variation.json"
    if batadal_var_file.exists():
        logger.info("Generating BATADAL parameter sensitivity plots...")
        with open(batadal_var_file) as f:
            batadal_variation = json.load(f)
        plot_parameter_sensitivity(
            batadal_variation,
            figures_path / "batadal_parameter_sensitivity.png",
        )
        plot_state_sensitivity(
            batadal_variation,
            figures_path / "batadal_state_sensitivity.png",
        )

    skab_file = results_path / "skab_experiments.json"
    if skab_file.exists():
        logger.info("Generating performance comparison plots...")
        with open(skab_file) as f:
            skab_data = json.load(f)

        # Çizimde kullanacağımız modellerin listesini çekiyoru
        models = list(skab_data[0]["original"].keys())
        models = [m for m in models if m not in {"automata_states", "automata_densities"}]

        metrics_list = []
        for seed_data in skab_data:
            seed = seed_data["seed"]
            supported_scenarios = {"original", "gaussian_noise", "unseen"}
            scenario_names = [s for s in seed_data.keys() if s in supported_scenarios]
            for scenario in scenario_names:
                for m in models:
                    if m not in seed_data[scenario]:
                        continue
                    mean_f1 = seed_data[scenario][m]["mean"]["f1"]
                    mean_acc = seed_data[scenario][m]["mean"]["accuracy"]
                    metrics_list.append(
                        {
                            "Seed": seed,
                            "Scenario": scenario,
                            "Model": m.upper(),
                            "F1 Score": mean_f1,
                            "Accuracy": mean_acc,
                        }
                    )

        df_metrics = pd.DataFrame(metrics_list)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_metrics, x="Scenario", y="F1 Score", hue="Model", palette="viridis", errorbar="sd")
        plt.title("Model F1 Performance Comparison across Scenarios (SKAB)", fontsize=14, fontweight="bold", pad=15)
        plt.ylabel("F1 Score", fontsize=12)
        plt.xlabel("Experiment Scenario", fontsize=12)
        plt.ylim(0, 1.05)
        plt.legend(frameon=True, facecolor="#111827", edgecolor="#374151")
        plt.tight_layout()
        plt.savefig(figures_path / "model_comparison_skab.png", dpi=300, facecolor="#0B0F19")
        plt.close()

        supported_scenarios = {"original", "gaussian_noise", "unseen"}
        for scenario in supported_scenarios:
            if scenario in skab_data[0]:
                emit_model_diagnostics("skab", skab_data[0][scenario], figures_path, scenario_suffix=scenario)

    batadal_file = results_path / "batadal_experiments.json"
    if batadal_file.exists():
        logger.info("Generating BATADAL diagnostic plots...")
        with open(batadal_file) as f:
            batadal_data = json.load(f)

        batadal_metrics_list = []
        supported_scenarios = {"original", "gaussian_noise", "unseen"}
        for seed_data in batadal_data:
            seed = seed_data["seed"]
            scenario_names = [s for s in seed_data.keys() if s in supported_scenarios]
            for scenario in scenario_names:
                scenario_payload = seed_data[scenario]
                models = [
                    key
                    for key in scenario_payload
                    if key not in {"automata_states", "automata_density", "automata_densities"}
                ]
                for model_name in models:
                    metrics = scenario_payload[model_name].get("metrics", {})
                    if "f1" not in metrics:
                        continue
                    batadal_metrics_list.append(
                        {
                            "Seed": seed,
                            "Scenario": scenario,
                            "Model": model_name.upper(),
                            "F1 Score": metrics["f1"],
                            "Accuracy": metrics.get("accuracy", 0.0),
                        }
                    )

        if batadal_metrics_list:
            df_batadal = pd.DataFrame(batadal_metrics_list)
            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=df_batadal,
                x="Scenario",
                y="F1 Score",
                hue="Model",
                palette="rocket",
                errorbar="sd",
            )
            plt.title(
                "Model F1 Performance Comparison across Scenarios (BATADAL)",
                fontsize=14,
                fontweight="bold",
                pad=15,
            )
            plt.ylabel("F1 Score", fontsize=12)
            plt.xlabel("Experiment Scenario", fontsize=12)
            plt.ylim(0, 1.05)
            plt.legend(frameon=True, facecolor="#111827", edgecolor="#374151")
            plt.tight_layout()
            plt.savefig(figures_path / "model_comparison_batadal.png", dpi=300, facecolor="#0B0F19")
            plt.close()

        for scenario in supported_scenarios:
            if scenario in batadal_data[0]:
                emit_model_diagnostics(
                    "batadal",
                    batadal_data[0][scenario],
                    figures_path,
                    scenario_suffix=scenario,
                )

    logger.info(f"Visualizations successfully generated and saved to {figures_path}")
