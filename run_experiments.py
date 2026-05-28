from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.config import load_config, ProjectConfig
from src.data.batadal_loader import (
    batadal_feature_columns,
    infer_batadal_target_column,
    load_batadal_training_dataset_2,
)
from src.data.skab_loader import load_skab, skab_feature_columns
from src.data.splits import (
    assert_no_group_leakage,
    make_batadal_chronological_split,
    make_skab_group_folds,
)
from src.evaluation.metrics import classification_metrics
from src.experiments.scenarios import add_gaussian_noise, inject_unseen_pattern
from src.models.deep_learning_pipeline import DeepLearningPipeline, set_seed
from src.pipeline import AutomataPipeline, build_fixed_automata_pipeline

# Log kayıtlarını konsola yazdırmak için yapılandırıyoruz
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_experiments")


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return classification_metrics(y_true, y_pred)


def run_skab_experiment(
    config: ProjectConfig,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    scenario: str,
    seed: int,
) -> dict[str, Any]:
    logger.info(f"Running SKAB experiment - Scenario: {scenario}, Seed: {seed}")
    set_seed(seed)

    # 1. Standart deney protokolü: source_file grubuna göre GroupKFold bölmesi yapıyoruz (sızıntıyı önlemek için)
    group_col = config.get("datasets", "skab", "group_column")
    n_splits = config.get("datasets", "skab", "n_splits", default=5)
    prefer_stratified = config.get("datasets", "skab", "prefer_stratified_group_kfold", default=True)

    splits = make_skab_group_folds(
        df,
        target_column=target_col,
        group_column=group_col,
        n_splits=n_splits,
        seed=seed,
        prefer_stratified=prefer_stratified,
    )

    models_to_run = ["automata"] + config.get("deep_learning", "models", default=["lstm", "gru"])
    results_by_model: dict[str, list[dict[str, float]]] = {m: [] for m in models_to_run}

    # Sonuç analizi için durum sayısını ve geçiş yoğunluğunu kaydedeceğimiz listeler
    automata_states = []
    automata_densities = []

    for fold_idx, split in enumerate(splits):
        assert_no_group_leakage(df, split, group_col)

        train_df = df.iloc[split.train].copy()
        test_df = df.iloc[split.test].copy()

        # Pencere boyutu (W) kadar öteleme yaparak test etiketlerini hizalıyoruz
        W = config.get("automata", "fixed_comparison", "window_size", default=4)
        y_test_aligned = test_df[target_col].to_numpy()[W - 1 :]

        # Gürültü senaryosu seçildiyse test verisine Gaussian gürültüsü ekliyoruz
        test_df_perturbed = test_df.copy()
        if scenario == "gaussian_noise":
            noise_std = config.get("experiments", "gaussian_noise", "std", default=0.05)
            noise_mean = config.get("experiments", "gaussian_noise", "mean", default=0.0)
            perturbed_vals = add_gaussian_noise(
                test_df_perturbed[feature_cols].to_numpy(),
                mean=noise_mean,
                std=noise_std,
                seed=seed + fold_idx,
            )
            test_df_perturbed[feature_cols] = perturbed_vals

        # --- A. Olasılıksal Otomata Modeli ---
        automata_pipe = build_fixed_automata_pipeline(config)
        automata_pipe.fit(train_df, feature_cols)

        # Unseen veri senaryosunda, SAX kelimelerini 'zzzz' gibi bilinmeyen bir pattern ile bozuyoruz
        if scenario == "unseen":
            test_patterns = automata_pipe.transform_patterns(test_df_perturbed)
            test_patterns_mutated = inject_unseen_pattern(test_patterns, replacement="zzzz")
            pred_sequence = automata_pipe.automata.predict_sequence(test_patterns_mutated)
            # Create a uniform prediction for comparison
            y_pred_automata = np.array([1 if pred_sequence["decision"] == "anomaly" else 0] * len(y_test_aligned))
        else:
            test_patterns = automata_pipe.transform_patterns(test_df_perturbed)
            y_pred_automata = []
            # Test aşamasında her adımda biriken olasılığa göre karar üretiyoruz
            for idx in range(1, len(test_patterns) + 1):
                decision_dict = automata_pipe.automata.predict_sequence(test_patterns[:idx])
                y_pred_automata.append(1 if decision_dict["decision"] == "anomaly" else 0)
            y_pred_automata = np.asarray(y_pred_automata)

        automata_metrics = evaluate_predictions(y_test_aligned, y_pred_automata)
        results_by_model["automata"].append(automata_metrics)
        automata_states.append(automata_pipe.automata.state_count)
        automata_densities.append(automata_pipe.automata.transition_density)

        # --- B. Derin Öğrenme Modelleri (LSTM, GRU vb.) ---
        for dl_model in [m for m in models_to_run if m != "automata"]:
            # Veri sızıntısını (leakage) önlemek için öznitelikleri sızıntısız normalleştiriyoruz
            from src.data.preprocessing import LeakageSafePreprocessor

            scaler = LeakageSafePreprocessor(use_standard_scaler=True, pca_components=None)
            train_X_scaled = scaler.fit_transform(train_df, feature_cols)
            test_X_scaled = scaler.transform(test_df_perturbed)

            train_y = train_df[target_col].to_numpy()
            test_y = test_df[target_col].to_numpy()

            # Eğitim verisinin son %10'unu kronolojik olarak validation (erken durdurma) için ayırıyoruz
            val_split = int(len(train_X_scaled) * 0.9)
            fit_train_X, fit_val_X = train_X_scaled[:val_split], train_X_scaled[val_split:]
            fit_train_y, fit_val_y = train_y[:val_split], train_y[val_split:]

            dl_pipe = DeepLearningPipeline(
                config=config,
                model_name=dl_model,
                input_size=len(feature_cols),
                window_size=W,
            )
            dl_pipe.fit(
                train_X=fit_train_X,
                train_y=fit_train_y,
                val_X=fit_val_X,
                val_y=fit_val_y,
            )

            # Modeli test seti üzerinde değerlendiriyoruz
            dl_metrics = dl_pipe.predict_and_evaluate(test_X_scaled, test_y)
            results_by_model[dl_model].append(dl_metrics)

    # Her bir model için fold ortalama ve standart sapmalarını çıkarıyoruz
    summary: dict[str, Any] = {}
    for m in models_to_run:
        metrics_df = pd.DataFrame(results_by_model[m])
        summary[m] = {
            "mean": metrics_df.mean().to_dict(),
            "std": metrics_df.std().to_dict(),
            "raw": results_by_model[m],
        }

    summary["automata_states"] = {
        "mean": float(np.mean(automata_states)),
        "std": float(np.std(automata_states)),
    }
    summary["automata_densities"] = {
        "mean": float(np.mean(automata_densities)),
        "std": float(np.std(automata_densities)),
    }
    return summary


def run_batadal_experiment(
    config: ProjectConfig,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    scenario: str,
    seed: int,
) -> dict[str, Any]:
    logger.info(f"Running BATADAL experiment - Scenario: {scenario}, Seed: {seed}")
    set_seed(seed)

    # Kronolojik veri bölme: %60 Eğitim, %20 Doğrulama, %20 Test
    split_conf = config.get("datasets", "batadal", "split")
    split = make_batadal_chronological_split(
        df,
        train_ratio=split_conf.get("train", 0.6),
        validation_ratio=split_conf.get("validation", 0.2),
        test_ratio=split_conf.get("test", 0.2),
    )

    train_df = df.iloc[split.train].copy()
    val_df = df.iloc[split.validation].copy()
    test_df = df.iloc[split.test].copy()

    # Zaman serisi pencere boyutu (W) kadar öteleyerek hedef etiketleri hizalıyoruz
    W = config.get("automata", "fixed_comparison", "window_size", default=4)
    y_test_aligned = test_df[target_col].to_numpy()[W - 1 :]

    # Gürültü ekleme senaryosunun kontrolü
    test_df_perturbed = test_df.copy()
    if scenario == "gaussian_noise":
        noise_std = config.get("experiments", "gaussian_noise", "std", default=0.05)
        noise_mean = config.get("experiments", "gaussian_noise", "mean", default=0.0)
        perturbed_vals = add_gaussian_noise(
            test_df_perturbed[feature_cols].to_numpy(),
            mean=noise_mean,
            std=noise_std,
            seed=seed,
        )
        test_df_perturbed[feature_cols] = perturbed_vals

    models_to_run = ["automata"] + config.get("deep_learning", "models", default=["lstm", "gru"])
    results_by_model: dict[str, dict[str, float]] = {}

    # --- A. Olasılıksal Otomata Modeli ---
    automata_pipe = build_fixed_automata_pipeline(config)
    # Veri sızıntısını önlemek için eğitim verisiyle fit ediyoruz
    automata_pipe.fit(train_df, feature_cols)

    if scenario == "unseen":
        test_patterns = automata_pipe.transform_patterns(test_df_perturbed)
        test_patterns_mutated = inject_unseen_pattern(test_patterns, replacement="zzzz")
        pred_sequence = automata_pipe.automata.predict_sequence(test_patterns_mutated)
        y_pred_automata = np.array([1 if pred_sequence["decision"] == "anomaly" else 0] * len(y_test_aligned))
    else:
        test_patterns = automata_pipe.transform_patterns(test_df_perturbed)
        y_pred_automata = []
        for idx in range(1, len(test_patterns) + 1):
            decision_dict = automata_pipe.automata.predict_sequence(test_patterns[:idx])
            y_pred_automata.append(1 if decision_dict["decision"] == "anomaly" else 0)
        y_pred_automata = np.asarray(y_pred_automata)

    results_by_model["automata"] = evaluate_predictions(y_test_aligned, y_pred_automata)

    # --- B. Derin Öğrenme Modelleri ---
    for dl_model in [m for m in models_to_run if m != "automata"]:
        from src.data.preprocessing import LeakageSafePreprocessor

        scaler = LeakageSafePreprocessor(use_standard_scaler=True, pca_components=None)
        train_X_scaled = scaler.fit_transform(train_df, feature_cols)
        val_X_scaled = scaler.transform(val_df)
        test_X_scaled = scaler.transform(test_df_perturbed)

        train_y = train_df[target_col].to_numpy()
        val_y = val_df[target_col].to_numpy()
        test_y = test_df[target_col].to_numpy()

        dl_pipe = DeepLearningPipeline(
            config=config,
            model_name=dl_model,
            input_size=len(feature_cols),
            window_size=W,
        )
        dl_pipe.fit(
            train_X=train_X_scaled,
            train_y=train_y,
            val_X=val_X_scaled,
            val_y=val_y,
        )

        results_by_model[dl_model] = dl_pipe.predict_and_evaluate(test_X_scaled, test_y)

    return {
        **results_by_model,
        "automata_states": automata_pipe.automata.state_count,
        "automata_density": automata_pipe.automata.transition_density,
    }


def run_automata_parameter_variation(
    config: ProjectConfig,
    skab_df: pd.DataFrame,
    skab_feature_cols: list[str],
    skab_target_col: str,
) -> list[dict[str, Any]]:
    logger.info("Running Automata Parameter Variation Analysis...")
    window_grid = config.get("automata", "parameter_grid", "window_size", default=[3, 4, 5, 6])
    alphabet_grid = config.get("automata", "parameter_grid", "alphabet_size", default=[3, 4, 5, 6])

    variation_results = []
    # Grid search analizinde tutarlılık sağlamak için sabit seed 42'yi kullanıyoruz
    set_seed(42)

    group_col = config.get("datasets", "skab", "group_column")
    n_splits = config.get("datasets", "skab", "n_splits", default=5)
    splits = make_skab_group_folds(
        skab_df,
        target_column=skab_target_col,
        group_column=group_col,
        n_splits=n_splits,
        seed=42,
    )

    for W in window_grid:
        for A in alphabet_grid:
            logger.info(f"Grid Search - window_size: {W}, alphabet_size: {A}")
            f1_scores = []
            state_counts = []
            transition_densities = []

            for split in splits:
                train_df = skab_df.iloc[split.train]
                test_df = skab_df.iloc[split.test]

                y_test_aligned = test_df[skab_target_col].to_numpy()[W - 1 :]

                # Belirlenen W ve A değerleriyle yeni bir Otomata hattı ayağa kaldırıyoruz
                pipe = AutomataPipeline(config=config, window_size=W, alphabet_size=A)
                pipe.fit(train_df, skab_feature_cols)

                test_patterns = pipe.transform_patterns(test_df)
                y_pred = []
                for idx in range(1, len(test_patterns) + 1):
                    decision_dict = pipe.automata.predict_sequence(test_patterns[:idx])
                    y_pred.append(1 if decision_dict["decision"] == "anomaly" else 0)

                metrics = evaluate_predictions(y_test_aligned, np.asarray(y_pred))
                f1_scores.append(metrics["f1"])
                state_counts.append(pipe.automata.state_count)
                transition_densities.append(pipe.automata.transition_density)

            variation_results.append(
                {
                    "window_size": W,
                    "alphabet_size": A,
                    "f1_mean": float(np.mean(f1_scores)),
                    "f1_std": float(np.std(f1_scores)),
                    "states_mean": float(np.mean(state_counts)),
                    "density_mean": float(np.mean(transition_densities)),
                }
            )

    return variation_results


def calculate_statistical_significance(skab_results_by_seed: list[dict[str, Any]]) -> dict[str, Any]:
    # F folds ve seeds üzerinden F1 skorlarını çıkararak Wilcoxon işaretli sıra testini yapıyoruz
    logger.info("Computing Wilcoxon signed-rank tests...")
    automata_f1 = []
    lstm_f1 = []
    gru_f1 = []

    for seed_res in skab_results_by_seed:
        # Check original scenario
        orig = seed_res["original"]
        for fold in orig["automata"]["raw"]:
            automata_f1.append(fold["f1"])
        for fold in orig["lstm"]["raw"]:
            lstm_f1.append(fold["f1"])
        for fold in orig["gru"]["raw"]:
            gru_f1.append(fold["f1"])

    sig_results = {}
    try:
        if len(automata_f1) >= 5:
            # Automata vs LSTM
            stat_al, p_val_al = wilcoxon(automata_f1, lstm_f1)
            sig_results["automata_vs_lstm"] = {"statistic": float(stat_al), "p_value": float(p_val_al)}

            # Automata vs GRU
            stat_ag, p_val_ag = wilcoxon(automata_f1, gru_f1)
            sig_results["automata_vs_gru"] = {"statistic": float(stat_ag), "p_value": float(p_val_ag)}
    except Exception as e:
        logger.warning(f"Could not compute Wilcoxon significance: {e}")
        sig_results["error"] = str(e)

    return sig_results


def main() -> None:
    config = load_config("config/config.yaml")
    results_dir = Path(config.get("paths", "results", default="outputs/results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Veri Setlerini Yüklüyoruz ---
    logger.info("Loading Datasets...")

    # A. SKAB
    skab_root = config.get("paths", "skab_root")
    skab_groups = config.get("datasets", "skab", "enabled_groups")
    skab_target = config.get("datasets", "skab", "target_column")
    skab_excluded = config.get("datasets", "skab", "excluded_input_columns")

    skab_df = load_skab(skab_root, skab_groups)
    skab_feature_cols = skab_feature_columns(skab_df, skab_target, skab_excluded)
    logger.info(f"SKAB loaded. Shape: {skab_df.shape}, Features: {len(skab_feature_cols)}")

    # B. BATADAL
    batadal_path = config.get("paths", "batadal_training_dataset_2")
    batadal_df = load_batadal_training_dataset_2(batadal_path)

    # BATADAL Training Dataset 2'deki normal çalışmayı temsil eden -999 değerlerini 0'a, anomalileri ise 1'e çekiyoruz
    if "ATT_FLAG" in batadal_df.columns:
        batadal_df["ATT_FLAG"] = batadal_df["ATT_FLAG"].replace(-999, 0).replace(2, 1)

    batadal_target = infer_batadal_target_column(batadal_df, config.get("datasets", "batadal", "target_column"))
    batadal_time_cols = config.get("datasets", "batadal", "time_columns")
    batadal_feature_cols = batadal_feature_columns(batadal_df, batadal_target, batadal_time_cols)
    logger.info(
        f"BATADAL loaded. Shape: {batadal_df.shape}, Target: {batadal_target}, Features: {len(batadal_feature_cols)}"
    )

    # --- 2. Deneyleri 5 Farklı Seed ve 3 Senaryo Altında Çalıştırıyoruz ---
    seeds = config.get("project", "random_seeds", default=[42, 123, 2026, 7, 999])
    scenarios = config.get("experiments", "scenarios", default=["original", "gaussian_noise", "unseen"])

    skab_full_results = []
    batadal_full_results = []

    for seed in seeds:
        skab_seed_res: dict[str, Any] = {}
        batadal_seed_res: dict[str, Any] = {}

        for scenario in scenarios:
            # Run SKAB
            skab_res = run_skab_experiment(
                config,
                skab_df,
                skab_feature_cols,
                skab_target,
                scenario,
                seed,
            )
            skab_seed_res[scenario] = skab_res

            # Run BATADAL
            batadal_res = run_batadal_experiment(
                config,
                batadal_df,
                batadal_feature_cols,
                batadal_target,
                scenario,
                seed,
            )
            batadal_seed_res[scenario] = batadal_res

        skab_full_results.append({"seed": seed, **skab_seed_res})
        batadal_full_results.append({"seed": seed, **batadal_seed_res})

    # --- 3. Modeller Arası İstatistiksel Anlamlılık Testi ---
    significance = calculate_statistical_significance(skab_full_results)

    # --- 4. Otomata Modelinde Parametre Analizi (Duyarlılık Testi) ---
    variation = run_automata_parameter_variation(
        config,
        skab_df,
        skab_feature_cols,
        skab_target,
    )

    # --- 5. Deney Sonuçlarını Dosyalara Kaydediyoruz ---
    logger.info("Saving results...")
    with open(results_dir / "skab_experiments.json", "w") as f:
        json.dump(skab_full_results, f, indent=2)

    with open(results_dir / "batadal_experiments.json", "w") as f:
        json.dump(batadal_full_results, f, indent=2)

    with open(results_dir / "statistical_significance.json", "w") as f:
        json.dump(significance, f, indent=2)

    with open(results_dir / "automata_parameter_variation.json", "w") as f:
        json.dump(variation, f, indent=2)

    logger.info(f"All experiments executed successfully. Outputs stored in {results_dir}")


if __name__ == "__main__":
    main()
