from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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


def plot_parameter_sensitivity(
    variation_data: list[dict[str, Any]],
    save_path: Path,
) -> None:
    # Pencere boyutu ve alfabe boyutunun F1 skoru üzerindeki duyarlılık grafiklerini çixiyoruz
    import pandas as pd

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
    import pandas as pd

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


def generate_all_visualizations(results_root: str | Path, figures_root: str | Path) -> None:
    results_path = Path(results_root)
    figures_path = Path(figures_root)
    figures_path.mkdir(parents=True, exist_ok=True)

    # 1. Aşama: Parametre Duyarlılık Çizimleri (Otomata modeli için grid arama sonuçları)
    var_file = results_path / "automata_parameter_variation.json"
    if var_file.exists():
        logger.info("Generating parameter sensitivity plots...")
        with open(var_file) as f:
            variation_data = json.load(f)
        plot_parameter_sensitivity(variation_data, figures_path / "parameter_sensitivity.png")
        plot_state_sensitivity(variation_data, figures_path / "state_sensitivity.png")

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
            for scenario in ["original", "gaussian_noise", "unseen"]:
                for m in models:
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

    logger.info(f"Visualizations successfully generated and saved to {figures_path}")
