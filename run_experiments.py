from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
from src.data.preprocessing import build_leakage_safe_preprocessor
from src.evaluation.metrics import classification_metrics
from src.evaluation.statistical_tests import calculate_statistical_significance
from src.evaluation.visualization import generate_all_visualizations
from src.experiments.experiment_logger import ExperimentLogger, build_experiment_record
from src.experiments.parameter_sweep import run_parameter_grid
from src.experiments.scenarios import add_gaussian_noise, inject_unseen_pattern
from src.models.deep_learning_pipeline import DeepLearningPipeline, set_seed
from src.pipeline import build_fixed_automata_pipeline

# Log kayıtlarını konsola yazdırmak için yapılandırıyoruz
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_experiments")

MAX_STORED_PREDICTIONS = 2_000
MAX_STORED_EXPLANATIONS = 25
EXPLANATIONS_DIR_NAME = "explanations"


def automata_experiment_parameters(config: ProjectConfig) -> dict[str, Any]:
    fixed = config.get("automata", "fixed_comparison")
    return {
        "window_size": fixed["window_size"],
        "alphabet_size": fixed["alphabet_size"],
    }


def export_automata_explanations(
    output_dir: Path,
    dataset: str,
    scenario: str,
    seed: int,
    samples: list[dict[str, Any]],
) -> None:
    if not samples:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{dataset}_{scenario}_seed{seed}_explanations.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(samples, handle, indent=2, ensure_ascii=False)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return classification_metrics(y_true, y_pred)


def anomaly_score(probability: float, threshold: float) -> float:
    if threshold <= 0:
        return 1.0 if probability <= threshold else 0.0
    return float(1.0 / (1.0 + (probability / threshold)))


def compact_prediction_records(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: list[float],
    window_size: int,
) -> list[dict[str, Any]]:
    limit = min(len(y_true), MAX_STORED_PREDICTIONS)
    return [
        {
            "time_step": int(idx + window_size - 1),
            "y_true": int(y_true[idx]),
            "y_pred": int(y_pred[idx]),
            "anomaly_score": float(scores[idx]),
        }
        for idx in range(limit)
    ]


def automata_predict_with_trace(
    automata_pipe: AutomataPipeline,
    df: pd.DataFrame,
    scenario: str,
    window_size: int,
) -> tuple[np.ndarray, list[float], list[dict[str, Any]], list[dict[str, Any]]]:
    patterns = automata_pipe.transform_patterns(df)
    if scenario == "unseen":
        
        patterns = inject_unseen_pattern(
            patterns,
            replacement="zzzz",
            forbidden_patterns=set(automata_pipe.automata.states_),
            alphabet_size=automata_pipe.alphabet_size,
        )

    # Not: naive prefix-skorlaması \(O(n^2)\) olduğu için (her prefix için yeniden olasılık hesaplayınca)
    # deney süresi saatlere uzayabiliyor. Burada geçiş olasılıklarını artımlı birikimli hesaplayarak
    # \(O(n)\) hale getiriyoruz.
    automata = automata_pipe.automata
    threshold = float(automata.threshold_ if automata.threshold_ is not None else automata.smoothing)

    mappings = [automata.map_pattern(p) for p in patterns]

    predictions: list[int] = []
    scores: list[float] = []
    explanations: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []

    path_probability = 1.0
    transitions_so_far: list[dict[str, Any]] = []

    for idx, current in enumerate(mappings):
        if idx == 0:
            path_probability = 1.0
        else:
            previous = mappings[idx - 1]
            prob = automata.transition_probability(previous.mapped, current.mapped)
            path_probability *= prob
            if len(transitions_so_far) < MAX_STORED_EXPLANATIONS:
                transitions_so_far.append(
                    {"from": previous.mapped, "to": current.mapped, "probability": float(prob)}
                )

        decision = "anomaly" if path_probability <= threshold else "normal"
        confidence = float(path_probability)

        predictions.append(1 if decision == "anomaly" else 0)
        scores.append(anomaly_score(float(path_probability), threshold))

        if len(explanations) < MAX_STORED_EXPLANATIONS:
            time_step = int(idx + window_size - 1)
            explanation = {
                "time_step": time_step,
                "state": current.mapped,
                "pattern": current.original,
                "status": current.status,
                "mapped_to": current.mapped if current.status == "unseen" else None,
                "mapping_distance": int(current.distance),
                "transitions": list(transitions_so_far),
                "probability": float(path_probability),
                "decision": decision,
                "confidence": confidence,
                "threshold": threshold,
            }
            explanations.append(explanation)
            trace.append(
                {
                    "time_step": time_step,
                    "state": explanation["state"],
                    "pattern": explanation["pattern"],
                    "status": explanation["status"],
                    "mapped_to": explanation["mapped_to"],
                    "probability": explanation["probability"],
                    "threshold": explanation["threshold"],
                    "decision": explanation["decision"],
                    "confidence": explanation["confidence"],
                }
            )

    return np.asarray(predictions), scores, trace, explanations


def run_skab_experiment(
    config: ProjectConfig,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    scenario: str,
    seed: int,
    models_override: list[str] | None = None,
    max_folds: int | None = None,
    experiment_logger: ExperimentLogger | None = None,
) -> dict[str, Any]:
    logger.info(f"Running SKAB experiment - Scenario: {scenario}, Seed: {seed}")
    set_seed(seed)

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

    models_to_run = models_override or (["automata"] + config.get("deep_learning", "models", default=["lstm", "gru"]))
    results_by_model: dict[str, list[dict[str, float]]] = {m: [] for m in models_to_run}
    outputs_by_model: dict[str, list[dict[str, Any]]] = {m: [] for m in models_to_run}

    automata_states = []
    automata_densities = []

    for fold_idx, split in enumerate(splits):
        if max_folds is not None and fold_idx >= max_folds:
            break
        logger.info(f"SKAB fold {fold_idx + 1}/{len(splits)} - Scenario: {scenario}, Seed: {seed}")
        assert_no_group_leakage(df, split, group_col)

        train_df = df.iloc[split.train].copy()
        test_df = df.iloc[split.test].copy()

        W = config.get("automata", "fixed_comparison", "window_size", default=4)
        y_test_aligned = test_df[target_col].to_numpy()[W - 1 :]

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

        logger.info("Fitting automata...")
        automata_pipe = build_fixed_automata_pipeline(config)
        automata_pipe.fit(train_df, feature_cols)

        logger.info("Predicting automata...")
        y_pred_automata, automata_scores, automata_trace, automata_explanations = automata_predict_with_trace(
            automata_pipe,
            test_df_perturbed,
            scenario=scenario,
            window_size=W,
        )

        automata_metrics = evaluate_predictions(y_test_aligned, y_pred_automata)
        results_by_model["automata"].append(automata_metrics)
        if experiment_logger is not None:
            experiment_logger.log(
                build_experiment_record(
                    dataset="skab",
                    model="automata",
                    scenario=scenario,
                    seed=seed,
                    parameters=automata_experiment_parameters(config),
                    metrics=automata_metrics,
                    fold=fold_idx,
                    extra={
                        "state_count": automata_pipe.automata.state_count,
                        "transition_density": automata_pipe.automata.transition_density,
                    },
                )
            )
        outputs_by_model["automata"].append(
            {
                "fold": fold_idx,
                "predictions": compact_prediction_records(y_test_aligned, y_pred_automata, automata_scores, W),
                "explanation_trace": automata_trace,
                "full_explanation_samples": automata_explanations,
                "transition_probabilities": automata_pipe.automata.transition_probabilities_,
            }
        )
        automata_states.append(automata_pipe.automata.state_count)
        automata_densities.append(automata_pipe.automata.transition_density)

        #  Derin Öğrenme Modelleri (LSTM, GRU vb.) 
        for dl_model in [m for m in models_to_run if m != "automata"]:
            logger.info(f"Training deep learning model: {dl_model} (fold {fold_idx + 1}/{len(splits)})")
            scaler = build_leakage_safe_preprocessor(config)
            train_X_scaled = scaler.fit_transform(train_df, feature_cols)
            test_X_scaled = scaler.transform(test_df_perturbed)

            train_y = train_df[target_col].to_numpy()
            test_y = test_df[target_col].to_numpy()

            # Eğitim verisinin son %10'unu kronolojik olarak validation için ayırıyoruz
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

            logger.info(f"Evaluating deep learning model: {dl_model} (fold {fold_idx + 1}/{len(splits)})")
            dl_probs = dl_pipe.predict_proba(test_X_scaled)
            dl_preds = (dl_probs >= 0.5).astype(int)
            dl_y_true = test_y[W - 1 :]
            dl_metrics = evaluate_predictions(dl_y_true, dl_preds)
            results_by_model[dl_model].append(dl_metrics)
            if experiment_logger is not None:
                dl_params = automata_experiment_parameters(config)
                dl_params["deep_learning_model"] = dl_model
                experiment_logger.log(
                    build_experiment_record(
                        dataset="skab",
                        model=dl_model,
                        scenario=scenario,
                        seed=seed,
                        parameters=dl_params,
                        metrics=dl_metrics,
                        fold=fold_idx,
                    )
                )
            outputs_by_model[dl_model].append(
                {
                    "fold": fold_idx,
                    "predictions": compact_prediction_records(dl_y_true, dl_preds, dl_probs.tolist(), W),
                }
            )

    # Her bir model için fold ortalama ve standart sapmalarını çıkarıyoruz
    summary: dict[str, Any] = {}
    for m in models_to_run:
        metrics_df = pd.DataFrame(results_by_model[m])
        
        std_series = metrics_df.std(ddof=0)
        std_series = std_series.fillna(0.0)
        summary[m] = {
            "mean": metrics_df.mean().to_dict(),
            "std": std_series.to_dict(),
            "raw": results_by_model[m],
            "outputs": outputs_by_model[m],
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
    models_override: list[str] | None = None,
    experiment_logger: ExperimentLogger | None = None,
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

    W = config.get("automata", "fixed_comparison", "window_size", default=4)
    y_test_aligned = test_df[target_col].to_numpy()[W - 1 :]

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

    models_to_run = models_override or (["automata"] + config.get("deep_learning", "models", default=["lstm", "gru"]))
    results_by_model: dict[str, dict[str, float]] = {}

    logger.info("Fitting automata...")
    automata_pipe = build_fixed_automata_pipeline(config)
    automata_pipe.fit(train_df, feature_cols)

    logger.info("Predicting automata...")
    y_pred_automata, automata_scores, automata_trace, automata_explanations = automata_predict_with_trace(
        automata_pipe,
        test_df_perturbed,
        scenario=scenario,
        window_size=W,
    )

    automata_metrics = evaluate_predictions(y_test_aligned, y_pred_automata)
    results_by_model["automata"] = {
        "metrics": automata_metrics,
        "predictions": compact_prediction_records(y_test_aligned, y_pred_automata, automata_scores, W),
        "explanation_trace": automata_trace,
        "full_explanation_samples": automata_explanations,
        "transition_probabilities": automata_pipe.automata.transition_probabilities_,
    }
    if experiment_logger is not None:
        experiment_logger.log(
            build_experiment_record(
                dataset="batadal",
                model="automata",
                scenario=scenario,
                seed=seed,
                parameters=automata_experiment_parameters(config),
                metrics=automata_metrics,
                extra={
                    "state_count": automata_pipe.automata.state_count,
                    "transition_density": automata_pipe.automata.transition_density,
                },
            )
        )

    for dl_model in [m for m in models_to_run if m != "automata"]:
        logger.info(f"Training deep learning model: {dl_model} (BATADAL)")
        scaler = build_leakage_safe_preprocessor(config)
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

        logger.info(f"Evaluating deep learning model: {dl_model} (BATADAL)")
        dl_probs = dl_pipe.predict_proba(test_X_scaled)
        dl_preds = (dl_probs >= 0.5).astype(int)
        dl_y_true = test_y[W - 1 :]
        dl_metrics = evaluate_predictions(dl_y_true, dl_preds)
        results_by_model[dl_model] = {
            "metrics": dl_metrics,
            "predictions": compact_prediction_records(dl_y_true, dl_preds, dl_probs.tolist(), W),
        }
        if experiment_logger is not None:
            dl_params = automata_experiment_parameters(config)
            dl_params["deep_learning_model"] = dl_model
            experiment_logger.log(
                build_experiment_record(
                    dataset="batadal",
                    model=dl_model,
                    scenario=scenario,
                    seed=seed,
                    parameters=dl_params,
                    metrics=dl_metrics,
                )
            )

    return {
        **results_by_model,
        "automata_states": automata_pipe.automata.state_count,
        "automata_density": automata_pipe.automata.transition_density,
    }


def run_skab_parameter_variation(
    config: ProjectConfig,
    skab_df: pd.DataFrame,
    skab_feature_cols: list[str],
    skab_target_col: str,
) -> list[dict[str, Any]]:
    logger.info("Running SKAB automata parameter variation...")
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
    train_test_splits = [(skab_df.iloc[split.train], skab_df.iloc[split.test]) for split in splits]
    return run_parameter_grid(
        config,
        dataset_name="skab",
        train_test_splits=train_test_splits,
        feature_columns=skab_feature_cols,
        target_column=skab_target_col,
    )


def run_batadal_parameter_variation(
    config: ProjectConfig,
    batadal_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> list[dict[str, Any]]:
    logger.info("Running BATADAL automata parameter variation...")
    split_conf = config.get("datasets", "batadal", "split")
    split = make_batadal_chronological_split(
        batadal_df,
        train_ratio=split_conf.get("train", 0.6),
        validation_ratio=split_conf.get("validation", 0.2),
        test_ratio=split_conf.get("test", 0.2),
    )
    train_df = batadal_df.iloc[split.train]
    test_df = batadal_df.iloc[split.test]
    return run_parameter_grid(
        config,
        dataset_name="batadal",
        train_test_splits=[(train_df, test_df)],
        feature_columns=feature_cols,
        target_column=target_col,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run explainable automata vs deep learning experiments.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Hızlı smoke-run: 1 seed, 1 senaryo (original), sadece automata. Geliştirme sırasında önerilir.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Virgülle ayrılmış seed listesi. Örn: 42,123. Boşsa config kullanılır.",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="",
        help='Virgülle ayrılmış senaryo listesi. Örn: original,gaussian_noise. Boşsa config kullanılır.',
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help='Virgülle ayrılmış model listesi. Örn: automata,lstm,gru,cnn1d. Boşsa config kullanılır.',
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=0,
        help="SKAB için en fazla kaç fold çalıştırılsın. 0 veya negatifse tüm fold'lar çalışır.",
    )
    args = parser.parse_args()

    config = load_config("config/config.yaml")
    results_dir = Path(config.get("paths", "results", default="outputs/results"))
    figures_dir = Path(config.get("paths", "figures", default="outputs/figures"))
    logs_dir = Path(config.get("paths", "logs", default="outputs/logs"))
    explanations_dir = results_dir.parent / EXPLANATIONS_DIR_NAME
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    experiment_logger = ExperimentLogger(logs_dir)
    if experiment_logger.jsonl_path.exists():
        experiment_logger.jsonl_path.unlink()

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

    # --- 2. Deneyleri Seed ve Senaryolar Altında Çalıştırıyoruz ---
    seeds = config.get("project", "random_seeds", default=[42, 123, 2026, 7, 999])
    scenarios = config.get("experiments", "scenarios", default=["original", "gaussian_noise", "unseen"])
    models_override: list[str] | None = None
    max_folds: int | None = None

    if args.fast:
        seeds = [42]
        scenarios = ["original"]
        models_override = ["automata"]
        max_folds = 1
        logger.info("FAST mode enabled: seeds=[42], scenarios=['original'], models=['automata']")

    if args.seeds.strip():
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        logger.info(f"CLI override: seeds={seeds}")

    if args.scenarios.strip():
        scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]
        logger.info(f"CLI override: scenarios={scenarios}")

    if args.models.strip():
        models_override = [x.strip() for x in args.models.split(",") if x.strip()]
        logger.info(f"CLI override: models={models_override}")

    if args.max_folds and args.max_folds > 0:
        max_folds = int(args.max_folds)
        logger.info(f"CLI override: max_folds={max_folds}")

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
                models_override=models_override,
                max_folds=max_folds,
                experiment_logger=experiment_logger,
            )
            skab_seed_res[scenario] = skab_res

            batadal_res = run_batadal_experiment(
                config,
                batadal_df,
                batadal_feature_cols,
                batadal_target,
                scenario,
                seed,
                models_override=models_override,
                experiment_logger=experiment_logger,
            )
            batadal_seed_res[scenario] = batadal_res

            automata_samples = batadal_res.get("automata", {}).get("full_explanation_samples", [])
            export_automata_explanations(explanations_dir, "batadal", scenario, seed, automata_samples)
            skab_automata_outputs = skab_res.get("automata", {}).get("outputs", [])
            if skab_automata_outputs:
                export_automata_explanations(
                    explanations_dir,
                    "skab",
                    scenario,
                    seed,
                    skab_automata_outputs[0].get("full_explanation_samples", []),
                )

        skab_full_results.append({"seed": seed, **skab_seed_res})
        batadal_full_results.append({"seed": seed, **batadal_seed_res})

    
    significance: dict[str, Any] = {}
    if not args.fast:
        significance = calculate_statistical_significance(skab_full_results)

    skab_variation: list[dict[str, Any]] = []
    batadal_variation: list[dict[str, Any]] = []
    if not args.fast:
        skab_variation = run_skab_parameter_variation(config, skab_df, skab_feature_cols, skab_target)
        batadal_variation = run_batadal_parameter_variation(
            config, batadal_df, batadal_feature_cols, batadal_target
        )

    logger.info("Saving results...")
    with open(results_dir / "skab_experiments.json", "w") as f:
        json.dump(skab_full_results, f, indent=2)

    with open(results_dir / "batadal_experiments.json", "w") as f:
        json.dump(batadal_full_results, f, indent=2)

    with open(results_dir / "statistical_significance.json", "w") as f:
        json.dump(significance, f, indent=2)

    with open(results_dir / "automata_parameter_variation.json", "w") as f:
        json.dump(skab_variation, f, indent=2)

    with open(results_dir / "batadal_parameter_variation.json", "w") as f:
        json.dump(batadal_variation, f, indent=2)

    experiment_logger.save_run_summary(
        {
            "seeds": seeds,
            "scenarios": scenarios,
            "models": models_override or config.get("deep_learning", "models", default=[]),
            "fast_mode": args.fast,
            "skab_target": skab_target,
            "batadal_target": batadal_target,
            "record_count": sum(1 for _ in experiment_logger.jsonl_path.open(encoding="utf-8"))
            if experiment_logger.jsonl_path.exists()
            else 0,
        }
    )

    if not args.fast:
        generate_all_visualizations(results_dir, figures_dir)
    else:
        logger.info("FAST mode: skipping full visualization generation.")
    logger.info(f"All experiments executed successfully. Outputs stored in {results_dir}")


if __name__ == "__main__":
    main()
