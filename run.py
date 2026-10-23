#!/usr/bin/env python3
"""Cost-aware credit default decisions: prediction-first versus decision-first.

The script downloads the UCI Default of Credit Card Clients file, fits four
probability models on the training split only, selects models and thresholds
on the validation split, and evaluates the locked rules once on the test split.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlrd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
UCI_ZIP = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)
UCI_XLS = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
    "default%20of%20credit%20card%20clients.xls"
)
XLS_NAME = "default of credit card clients.xls"
TARGET = "default payment next month"
SEED = 42
SPLIT_SEEDS = tuple(range(42, 62))
RATIOS = (2, 5, 10, 20)
N_BOOT = 1000
THRESHOLDS = np.array([i / 100 for i in range(1, 100)], dtype=np.float64)

DISPLAY = {
    "LogisticRegression": "logistic regression",
    "RandomForest": "random forest",
    "HistGradientBoosting": "histogram GB",
    "XGBoost": "XGBoost",
}
MODEL_ORDER = (
    "LogisticRegression",
    "RandomForest",
    "HistGradientBoosting",
    "XGBoost",
)

# Okabe-Ito, colorblind-safe.
BLUE = "#0072B2"
VERMILION = "#D55E00"
GREEN = "#009E73"
GRAY = "#666666"


def fmt4(value: float) -> str:
    return f"{float(value):.4f}"


def fmt2(value: float) -> str:
    return f"{float(value):.2f}"


def download_xls() -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / XLS_NAME
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using cached data file {dest}", flush=True)
        return dest
    zip_path = CACHE / "uci350.zip"
    print(f"Downloading {UCI_ZIP}", flush=True)
    request = urllib.request.Request(UCI_ZIP, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            zip_path.write_bytes(response.read())
        with zipfile.ZipFile(zip_path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".xls")]
            if len(members) != 1:
                raise RuntimeError(f"Expected one xls in the UCI zip, found {members}")
            dest.write_bytes(archive.read(members[0]))
    except Exception as exc:
        print(f"Zip download failed ({exc}); trying the direct xls URL.", flush=True)
        request = urllib.request.Request(UCI_XLS, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            dest.write_bytes(response.read())
    if not dest.exists() or dest.stat().st_size == 0:
        raise SystemExit("UCI download did not produce a workbook.")
    return dest


def load_frame(path: Path) -> tuple[pd.DataFrame, dict]:
    book = xlrd.open_workbook(path)
    sheet = book.sheet_by_index(0)
    header = [str(sheet.cell_value(1, col)).strip() for col in range(sheet.ncols)]
    if header[0] != "ID" or "SEX" not in header or header[-1] != TARGET:
        raise SystemExit(f"Unexpected UCI header row: {header}")
    records = [
        [sheet.cell_value(row, col) for col in range(sheet.ncols)]
        for row in range(2, sheet.nrows)
    ]
    frame = pd.DataFrame(records, columns=header)
    if len(frame) != 30000:
        raise SystemExit(f"Expected 30000 data rows, found {len(frame)}.")
    if frame.isna().any().any() or (frame == "").any().any():
        raise SystemExit("The UCI file has missing values; this script does not impute.")
    sex_present = "SEX" in frame.columns
    frame = frame.drop(columns=["ID", "SEX"])
    if TARGET not in frame.columns or "SEX" in frame.columns or "ID" in frame.columns:
        raise SystemExit("ID or SEX remained in the modeling table, or the target is missing.")
    y = frame[TARGET].astype(int)
    if set(y.unique()) != {0, 1}:
        raise SystemExit(f"Target is not binary 0/1: {sorted(y.unique())}")
    predictors = [column for column in frame.columns if column != TARGET]
    x = frame[predictors].astype(float)
    if x.shape[1] != 22:
        raise SystemExit(f"Expected 22 predictors after dropping ID and SEX, found {x.shape[1]}.")
    default_rate = float(y.mean())
    if not (0.20 <= default_rate <= 0.24):
        raise SystemExit(f"Default rate {default_rate:.4f} is not about 22%.")
    meta = {
        "n_rows": int(len(frame)),
        "n_defaults": int(y.sum()),
        "default_rate": default_rate,
        "sex_present_in_source": bool(sex_present),
        "dropped_columns": ["ID", "SEX"],
        "predictors": predictors,
        "n_predictors": len(predictors),
        "target": TARGET,
        "source_url": UCI_ZIP,
        "dataset_doi": "10.24432/C55S3H",
    }
    return x, y, meta


def build_models() -> dict:
    return {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        max_iter=2000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            max_features="sqrt",
            bootstrap=True,
            random_state=SEED,
            n_jobs=1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=6,
            learning_rate=0.1,
            early_stopping=False,
            random_state=SEED,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        ),
    }


def positive_proba(model, features: np.ndarray) -> np.ndarray:
    estimator = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    classes = np.asarray(estimator.classes_)
    if list(classes.tolist()) != [0, 1]:
        raise SystemExit(f"Expected classes [0, 1], found {classes.tolist()}.")
    proba = model.predict_proba(features)
    return proba[:, 1].astype(np.float64)


def counts_at_thresholds(y: np.ndarray, proba: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(y).astype(np.int8)
    predicted = proba[:, None] >= THRESHOLDS[None, :]
    positive = labels[:, None] == 1
    true_positive = np.count_nonzero(predicted & positive, axis=0).astype(np.int64)
    false_positive = np.count_nonzero(predicted & ~positive, axis=0).astype(np.int64)
    false_negative = np.count_nonzero(~predicted & positive, axis=0).astype(np.int64)
    return true_positive, false_positive, false_negative


def f1_scores(true_positive: np.ndarray, false_positive: np.ndarray, false_negative: np.ndarray) -> np.ndarray:
    tp = true_positive.astype(np.float64)
    fp = false_positive.astype(np.float64)
    fn = false_negative.astype(np.float64)
    precision = np.zeros_like(tp)
    recall = np.zeros_like(tp)
    np.divide(tp, tp + fp, out=precision, where=(tp + fp) > 0)
    np.divide(tp, tp + fn, out=recall, where=(tp + fn) > 0)
    scores = np.zeros_like(tp)
    np.divide(2 * precision * recall, precision + recall, out=scores, where=(precision + recall) > 0)
    return scores


def index_of_half() -> int:
    matches = np.where(np.isclose(THRESHOLDS, 0.50))[0]
    if len(matches) != 1:
        raise SystemExit("Threshold grid does not contain 0.50 exactly once.")
    return int(matches[0])


def choose_threshold(numerator: np.ndarray, f1: np.ndarray) -> int:
    """Minimize cost, then maximize F1, then take the lower threshold."""
    best_index = 0
    best_key = None
    for index, threshold in enumerate(THRESHOLDS):
        key = (int(numerator[index]), -float(f1[index]), float(threshold))
        if best_key is None or key < best_key:
            best_key = key
            best_index = index
    return best_index


def threshold_predictions(proba: np.ndarray, threshold: float) -> np.ndarray:
    return (proba >= threshold).astype(np.int8)


def paired_cost_difference(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    ratio: int,
    resample_index: np.ndarray,
) -> tuple[float, float, bool]:
    labels = y[resample_index]
    cost_a = _resample_cost(labels, pred_a[resample_index], ratio)
    cost_b = _resample_cost(labels, pred_b[resample_index], ratio)
    difference = cost_a - cost_b
    low, high = np.percentile(difference, [2.5, 97.5], method="linear")
    includes_zero = bool(low <= 0.0 <= high)
    return float(low), float(high), includes_zero


def _resample_cost(labels: np.ndarray, predicted: np.ndarray, ratio: int) -> np.ndarray:
    false_positive = np.count_nonzero((predicted == 1) & (labels == 0), axis=1)
    false_negative = np.count_nonzero((predicted == 0) & (labels == 1), axis=1)
    return (false_positive + ratio * false_negative) / labels.shape[1]


def point_cost(y: np.ndarray, predicted: np.ndarray, ratio: int) -> float:
    labels = np.asarray(y).astype(np.int8)
    decisions = np.asarray(predicted).astype(np.int8)
    false_positive = int(np.count_nonzero((decisions == 1) & (labels == 0)))
    false_negative = int(np.count_nonzero((decisions == 0) & (labels == 1)))
    return float((false_positive + ratio * false_negative) / labels.shape[0])


RATIO_STYLE = {
    2: ("#0072B2", "-"),
    5: ("#E69F00", "--"),
    10: ("#009E73", "-."),
    20: ("#D55E00", ":"),
}


def save_figure_1(path: Path, results: dict) -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.6,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9 / 2.54, 7.6 / 2.54), sharex=True)
    panel = {2: "a", 5: "b", 10: "c", 20: "d"}
    for axis, ratio in zip(axes.ravel(), RATIOS):
        curve = np.asarray(results["by_ratio"][str(ratio)]["strategy_a"]["validation_cost_curve"], dtype=float)
        color, style = RATIO_STYLE[ratio]
        chosen = float(results["by_ratio"][str(ratio)]["strategy_a"]["threshold"])
        axis.plot(THRESHOLDS, curve, color=color, linestyle=style, linewidth=1.15)
        chosen_index = int(np.where(np.isclose(THRESHOLDS, chosen))[0][0])
        chosen_cost = float(curve[chosen_index])
        axis.plot([chosen], [chosen_cost], marker="o", color=color, markersize=4.8, zorder=4)
        y_span = float(curve.max() - curve.min())
        axis.text(
            min(chosen + 0.04, 0.86),
            chosen_cost + 0.08 * y_span,
            f"{chosen:.2f}",
            color=color,
            fontsize=6.2,
            ha="left",
            va="bottom",
            clip_on=False,
            zorder=5,
        )
        axis.axvline(0.50, color=GRAY, linestyle=":", linewidth=0.7)
        # r = 2 is high on the left, so its label sits on the right.
        label_x = 0.52 if ratio == 2 else 0.05
        axis.text(label_x, 0.86, f"({panel[ratio]})  r = {ratio}", transform=axis.transAxes, fontsize=7)
        axis.set_xlim(0.0, 1.0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#dddddd", linewidth=0.4)
    axes[0, 0].set_ylabel("Val. cost")
    axes[1, 0].set_ylabel("Val. cost")
    axes[1, 0].set_xlabel("Threshold")
    axes[1, 1].set_xlabel("Threshold")
    fig.tight_layout(h_pad=0.45, w_pad=0.35)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def save_figure_2(path: Path, results: dict) -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
        }
    )
    ratios = [int(ratio) for ratio in results["by_ratio"]]
    positions = np.arange(len(ratios))
    summary = results["repeat_summary"]["by_ratio"]

    def series(key: str) -> np.ndarray:
        return np.array([summary[str(ratio)][key]["seed42"] for ratio in ratios], dtype=float)

    def whisker(key: str) -> np.ndarray:
        primary = series(key)
        low = np.array([summary[str(ratio)][key]["min"] for ratio in ratios], dtype=float)
        high = np.array([summary[str(ratio)][key]["max"] for ratio in ratios], dtype=float)
        return np.vstack([primary - low, high - primary])

    threshold_a = series("threshold_a")
    threshold_b = series("threshold_b")
    cost_a = series("test_cost_a")
    cost_b = series("test_cost_b")
    cost_half = series("test_cost_at_0.50")

    fig, axes = plt.subplots(2, 1, figsize=(9 / 2.54, 8.2 / 2.54), sharex=True)
    axes[0].errorbar(positions, threshold_a, yerr=whisker("threshold_a"), color=BLUE, marker="o", linestyle="-", label="A prediction-first", linewidth=1.2, markersize=4.5, capsize=2, elinewidth=0.7)
    axes[0].errorbar(positions, threshold_b, yerr=whisker("threshold_b"), color=VERMILION, marker="s", linestyle="--", label="B decision-first", linewidth=1.2, markersize=4.5, capsize=2, elinewidth=0.7)
    axes[0].axhline(0.50, color=GRAY, linestyle=":", linewidth=0.9, label="threshold 0.50")
    threshold_limits = list(threshold_a) + list(threshold_b) + [0.50]
    threshold_limits += [summary[str(ratio)]["threshold_a"]["min"] for ratio in ratios]
    threshold_limits += [summary[str(ratio)]["threshold_b"]["max"] for ratio in ratios]
    axes[0].set_ylim(min(threshold_limits) - 0.05, 0.64)
    axes[0].set_ylabel("Threshold")
    axes[0].text(0.02, 0.90, "(a)", transform=axes[0].transAxes, fontsize=8, fontweight="bold")
    axes[0].legend(frameon=False, loc="upper right", handlelength=1.6, borderaxespad=0.2)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.4)

    axes[1].errorbar(positions, cost_a, yerr=whisker("test_cost_a"), color=BLUE, marker="o", linestyle="-", label="A prediction-first", linewidth=1.2, markersize=4.5, capsize=2, elinewidth=0.7)
    axes[1].errorbar(positions, cost_b, yerr=whisker("test_cost_b"), color=VERMILION, marker="s", linestyle="--", label="B decision-first", linewidth=1.2, markersize=4.5, capsize=2, elinewidth=0.7)
    axes[1].errorbar(positions, cost_half, yerr=whisker("test_cost_at_0.50"), color=GREEN, marker="x", linestyle=":", label="A at 0.50", linewidth=1.2, markersize=5, capsize=2, elinewidth=0.7)
    axes[1].set_ylabel("Test cost")
    axes[1].set_xlabel("Cost ratio r")
    axes[1].text(0.02, 0.90, "(b)", transform=axes[1].transAxes, fontsize=8, fontweight="bold")
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.14, 1.0), handlelength=1.6, borderaxespad=0.1)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.4)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([str(ratio) for ratio in ratios])
    fig.tight_layout(h_pad=0.8)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def assert_vector_pdf(path: Path) -> None:
    data = path.read_bytes()
    if not data.startswith(b"%PDF"):
        raise SystemExit(f"{path.name} is not a PDF.")
    if b"/Subtype /Image" in data or b"/Subtype/Image" in data:
        raise SystemExit(f"{path.name} contains a raster image.")
    if b"/Font" not in data:
        raise SystemExit(f"{path.name} does not embed a font.")


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def analyze_split(frame_x: pd.DataFrame, frame_y: pd.Series, seed: int) -> dict:
    half = index_of_half()
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        frame_x, frame_y, test_size=6000, stratify=frame_y, random_state=seed, shuffle=True
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=6000, stratify=y_trainval, random_state=seed, shuffle=True
    )
    if (len(x_train), len(x_val), len(x_test)) != (18000, 6000, 6000):
        raise SystemExit("Split sizes were not 18000/6000/6000.")

    x_train_np = x_train.to_numpy(dtype=np.float64)
    y_train_np = y_train.to_numpy(dtype=np.int8)
    x_val_np = x_val.to_numpy(dtype=np.float64)
    y_val_np = y_val.to_numpy(dtype=np.int8)
    x_test_np = x_test.to_numpy(dtype=np.float64)
    y_test_np = y_test.to_numpy(dtype=np.int8)

    fitted = {}
    val_proba = {}
    val_counts = {}
    for name, model in build_models().items():
        if seed == SEED:
            print(f"Fitting {name}", flush=True)
        model.fit(x_train_np, y_train_np)
        fitted[name] = model
        proba = positive_proba(model, x_val_np)
        val_proba[name] = proba
        val_counts[name] = counts_at_thresholds(y_val_np, proba)

    f1_at_half = {}
    for name in MODEL_ORDER:
        true_positive, false_positive, false_negative = val_counts[name]
        f1_at_half[name] = float(f1_scores(true_positive, false_positive, false_negative)[half])
    best_f1 = max(f1_at_half.values())
    tied = [name for name in MODEL_ORDER if f1_at_half[name] == best_f1]
    tie_break_used = len(tied) > 1

    validation_cache = {}
    for name in MODEL_ORDER:
        true_positive, false_positive, false_negative = val_counts[name]
        validation_cache[name] = {
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "f1": f1_scores(true_positive, false_positive, false_negative),
        }

    def model_for_ratio(ratio: int) -> str:
        if not tie_break_used:
            return tied[0]
        return min(
            tied,
            key=lambda name: (
                int(validation_cache[name]["fp"][half] + ratio * validation_cache[name]["fn"][half]),
                name,
            ),
        )

    by_ratio = {}
    chosen_models = set()
    for ratio in RATIOS:
        model_a = model_for_ratio(ratio)
        cache_a = validation_cache[model_a]
        numer_a = cache_a["fp"] + ratio * cache_a["fn"]
        index_a = choose_threshold(numer_a, cache_a["f1"])
        best_b = None
        best_key = None
        for name in MODEL_ORDER:
            cache = validation_cache[name]
            numer = cache["fp"] + ratio * cache["fn"]
            for index, threshold in enumerate(THRESHOLDS):
                key = (int(numer[index]), -float(cache["f1"][index]), name, float(threshold))
                if best_key is None or key < best_key:
                    best_key = key
                    best_b = (name, index)
        model_b, index_b = best_b
        chosen_models.add(model_a)
        chosen_models.add(model_b)

        def pack(name: str, index: int) -> dict:
            cache = validation_cache[name]
            threshold = float(THRESHOLDS[index])
            numer = int(cache["fp"][index] + ratio * cache["fn"][index])
            packed = {
                "model": name,
                "display_name": DISPLAY[name],
                "threshold": threshold,
                "validation_cost": numer / len(y_val_np),
                "validation_f1": float(cache["f1"][index]),
                "validation_fp": int(cache["fp"][index]),
                "validation_fn": int(cache["fn"][index]),
            }
            if seed == SEED and name == model_a:
                packed["validation_cost_curve"] = (
                    (cache["fp"] + ratio * cache["fn"]) / len(y_val_np)
                ).astype(float).tolist()
            return packed

        by_ratio[str(ratio)] = {
            "strategy_a": pack(model_a, index_a),
            "strategy_b": pack(model_b, index_b),
        }

    # Test probabilities are computed only after every selection is locked.
    test_proba = {}
    for name in sorted(chosen_models):
        if seed == SEED:
            print(f"Scoring test split with {name}", flush=True)
        test_proba[name] = positive_proba(fitted[name], x_test_np)

    resample_index = None
    if seed == SEED:
        resample_index = np.random.default_rng(SEED).integers(0, len(y_test_np), size=(N_BOOT, len(y_test_np)))

    fixed_models = {row["strategy_a"]["model"] for row in by_ratio.values()}
    prediction_first_model = next(iter(fixed_models)) if len(fixed_models) == 1 else None

    for ratio in RATIOS:
        row = by_ratio[str(ratio)]
        rule_a = row["strategy_a"]
        rule_b = row["strategy_b"]
        pred_a = threshold_predictions(test_proba[rule_a["model"]], rule_a["threshold"])
        pred_b = threshold_predictions(test_proba[rule_b["model"]], rule_b["threshold"])
        pred_half = threshold_predictions(test_proba[rule_a["model"]], 0.50)
        cost_a = point_cost(y_test_np, pred_a, ratio)
        cost_b = point_cost(y_test_np, pred_b, ratio)
        cost_half = point_cost(y_test_np, pred_half, ratio)
        if cost_a == 0:
            raise SystemExit(f"Test cost for strategy A at r={ratio} is zero; relative difference is undefined.")
        rule_a["test_cost"] = cost_a
        rule_b["test_cost"] = cost_b
        row["cost_difference"] = cost_a - cost_b
        row["relative_difference"] = (cost_a - cost_b) / cost_a
        row["threshold_effect"] = {
            "model": rule_a["model"],
            "display_name": rule_a["display_name"],
            "test_cost_at_0.50": cost_half,
            "tuned_threshold": rule_a["threshold"],
            "test_cost_at_tuned_threshold": cost_a,
            "difference_0.50_minus_tuned": cost_half - cost_a,
        }
        if resample_index is not None:
            low, high, includes_zero = paired_cost_difference(y_test_np, pred_a, pred_b, ratio, resample_index)
            effect_low, effect_high, effect_includes = paired_cost_difference(
                y_test_np, pred_half, pred_a, ratio, resample_index
            )
            row["bootstrap"] = {
                "n_resamples": N_BOOT,
                "seed": SEED,
                "method": "paired percentile",
                "interval": "2.5 to 97.5",
                "contrast": "test cost A minus test cost B",
                "ci_low": low,
                "ci_high": high,
                "includes_zero": includes_zero,
                "retrained": False,
            }
            row["threshold_effect"]["bootstrap"] = {
                "contrast": "test cost at 0.50 minus test cost at the tuned threshold",
                "ci_low": effect_low,
                "ci_high": effect_high,
                "includes_zero": effect_includes,
                "same_resamples_as_strategy_contrast": True,
            }

    compact = {
        "seed": seed,
        "prediction_first_model": prediction_first_model,
        "tie_break_used": tie_break_used,
        "by_ratio": {
            str(ratio): {
                "a_model": by_ratio[str(ratio)]["strategy_a"]["model"],
                "a_threshold": by_ratio[str(ratio)]["strategy_a"]["threshold"],
                "a_test_cost": by_ratio[str(ratio)]["strategy_a"]["test_cost"],
                "b_model": by_ratio[str(ratio)]["strategy_b"]["model"],
                "b_threshold": by_ratio[str(ratio)]["strategy_b"]["threshold"],
                "b_test_cost": by_ratio[str(ratio)]["strategy_b"]["test_cost"],
                "cost_at_0.50": by_ratio[str(ratio)]["threshold_effect"]["test_cost_at_0.50"],
                "relative_difference": by_ratio[str(ratio)]["relative_difference"],
            }
            for ratio in RATIOS
        },
    }
    if seed != SEED:
        return {"compact": compact, "full": None}

    results = {
        "dataset": None,
        "split": {
            "seed": seed,
            "stratified": True,
            "n_train": 18000,
            "n_validation": 6000,
            "n_test": 6000,
            "train_default_rate": float(y_train_np.mean()),
            "validation_default_rate": float(y_val_np.mean()),
            "test_default_rate": float(y_test_np.mean()),
            "procedure": "Hold out 6000 test rows, then hold out 6000 validation rows from the remainder.",
        },
        "hyperparameters": {
            "class_weight": None,
            "logistic_regression": {
                "pipeline": "StandardScaler then LogisticRegression",
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 2000,
                "penalty": "library default L2",
            },
            "random_forest": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
            },
            "hist_gradient_boosting": {
                "max_iter": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "early_stopping": False,
            },
            "xgboost": {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.1,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "scale_pos_weight": "not set",
            },
            "random_state": SEED,
            "n_jobs": 1,
        },
        "threshold_grid": {"min": 0.01, "max": 0.99, "step": 0.01, "n": int(THRESHOLDS.size)},
        "validation_f1_at_0.50": f1_at_half,
        "prediction_first": {
            "tied_models": tied,
            "tie_break_used": tie_break_used,
            "fixed_across_ratios": prediction_first_model is not None,
            "model": prediction_first_model,
            "display_name": None if prediction_first_model is None else DISPLAY[prediction_first_model],
            "validation_f1_at_0.50": None if prediction_first_model is None else f1_at_half[prediction_first_model],
            "tie_break_rule": (
                "Highest validation F1 at threshold 0.50. "
                "If F1 ties, lower validation cost at threshold 0.50 for that r, then model name."
            ),
        },
        "selection_rules": {
            "cost": "(1 * FP + r * FN) / N",
            "positive_class": "default payment next month = 1",
            "decision": "predict default if probability >= threshold",
            "strategy_a_threshold_tie_break": "lower cost, then higher validation F1, then lower threshold",
            "strategy_b_tie_break": "lower validation cost, then higher validation F1, then model name, then lower threshold",
            "model_name_order": list(MODEL_ORDER),
        },
        "by_ratio": by_ratio,
        "bootstrap": {
            "n_resamples": N_BOOT,
            "seed": SEED,
            "method": "paired percentile",
            "retrained": False,
        },
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": __import__("sklearn").__version__,
            "xgboost": __import__("xgboost").__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    return {"compact": compact, "full": results}


def _range(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {"min": float(array.min()), "median": float(np.median(array)), "max": float(array.max())}


def run_experiment() -> dict:
    started = time.perf_counter()
    if not np.array_equal(THRESHOLDS, np.round(THRESHOLDS, 2)):
        raise SystemExit("Threshold grid is not on the 0.01 lattice.")
    if SEED not in SPLIT_SEEDS:
        raise SystemExit("The primary seed is not in the repeated-split list.")
    frame_x, frame_y, meta = load_frame(download_xls())
    compacts = []
    primary = None
    for seed in SPLIT_SEEDS:
        print(f"Split seed {seed}", flush=True)
        outcome = analyze_split(frame_x, frame_y, seed)
        compacts.append(outcome["compact"])
        if seed == SEED:
            primary = outcome["full"]
    if primary is None:
        raise SystemExit("Primary split was not evaluated.")
    summary = {}
    for ratio in RATIOS:
        rows = [item["by_ratio"][str(ratio)] for item in compacts]
        primary_row = next(item["by_ratio"][str(ratio)] for item in compacts if item["seed"] == SEED)

        def pack(key: str, primary_key: str) -> dict:
            stats = _range([row[key] for row in rows])
            stats["seed42"] = primary_row[primary_key]
            return stats

        tuned_lower = sum(row["cost_at_0.50"] > row["a_test_cost"] for row in rows)
        decision_lower = sum(row["b_test_cost"] < row["a_test_cost"] for row in rows)
        summary[str(ratio)] = {
            "threshold_a": pack("a_threshold", "a_threshold"),
            "threshold_b": pack("b_threshold", "b_threshold"),
            "test_cost_a": pack("a_test_cost", "a_test_cost"),
            "test_cost_b": pack("b_test_cost", "b_test_cost"),
            "test_cost_at_0.50": pack("cost_at_0.50", "cost_at_0.50"),
            "n_seeds_tuned_cost_below_0.50": int(tuned_lower),
            "n_seeds_decision_first_lower_cost": int(decision_lower),
            "prediction_first_models": sorted({item["prediction_first_model"] for item in compacts}),
        }
    model_counts = {}
    for item in compacts:
        model_counts[item["prediction_first_model"]] = model_counts.get(item["prediction_first_model"], 0) + 1
    primary["dataset"] = meta
    primary["split"]["n_repeated_splits"] = len(SPLIT_SEEDS)
    primary["split"]["repeated_seeds"] = list(SPLIT_SEEDS)
    primary["split"]["model_random_state"] = SEED
    primary["repeat_summary"] = {
        "n_seeds": len(SPLIT_SEEDS),
        "seeds": list(SPLIT_SEEDS),
        "prediction_first_model_counts": model_counts,
        "by_ratio": summary,
        "note": "Min, median, and max are over split seeds. They are not a confidence interval. The bootstrap interval is only for seed 42.",
    }
    primary["repeats"] = compacts
    primary["elapsed_seconds"] = time.perf_counter() - started
    return json_ready(primary)


def require(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


def check_manuscript(results: dict) -> None:
    path = ROOT / "manuscript.tex"
    if not path.exists():
        print("manuscript.tex is not present; number check skipped.", flush=True)
        return
    tex = path.read_text(encoding="utf-8")
    errors: list[str] = []

    def table_block(label: str) -> str:
        match = re.search(
            rf"\\begin\{{table\}}.*?\\label\{{{label}\}}.*?\\end\{{table\}}",
            tex,
            flags=re.S,
        )
        if match is None:
            errors.append(f"Missing table labeled {label}.")
            return ""
        return match.group(0)

    validation = table_block("tab:validation")
    test = table_block("tab:test")
    validation_rows = re.findall(
        r"(?m)^\s*(\d+)\s*&\s*([AB])\s*&\s*([^&]+?)\s*&\s*(-?\d+\.\d+)\s*&\s*(-?\d+\.\d+)\s*&\s*(-?\d+\.\d+)\s*\\\\",
        validation,
    )
    test_rows = re.findall(
        r"(?m)^\s*(\d+)\s*&\s*([AB])\s*&\s*([^&]+?)\s*&\s*(-?\d+\.\d+)\s*&\s*(-?\d+\.\d+)\s*\\\\",
        test,
    )
    expected_validation = []
    expected_test = []
    for ratio in RATIOS:
        row = results["by_ratio"][str(ratio)]
        for letter, key in (("A", "strategy_a"), ("B", "strategy_b")):
            rule = row[key]
            expected_validation.append(
                (
                    str(ratio),
                    letter,
                    rule["display_name"],
                    fmt2(rule["threshold"]),
                    fmt4(rule["validation_cost"]),
                    fmt4(rule["validation_f1"]),
                )
            )
            expected_test.append(
                (
                    str(ratio),
                    letter,
                    rule["display_name"],
                    fmt2(rule["threshold"]),
                    fmt4(rule["test_cost"]),
                )
            )
    require(
        validation_rows == expected_validation,
        errors,
        f"Table 1 rows do not match results.json.\nParsed: {validation_rows}\nExpected: {expected_validation}",
    )
    require(
        test_rows == expected_test,
        errors,
        f"Table 2 rows do not match results.json.\nParsed: {test_rows}\nExpected: {expected_test}",
    )

    ci_matches = re.findall(
        r"\(A-B\)/A = (-?\d+\.\d+)\$\. Cost difference \$A-B = (-?\d+\.\d+)\$, "
        r"95\\% CI \$\[(-?\d+\.\d+),\\ (-?\d+\.\d+)\]\$, includes zero: (yes|no)\.",
        test,
    )
    expected_ci = []
    for ratio in RATIOS:
        row = results["by_ratio"][str(ratio)]
        expected_ci.append(
            (
                fmt4(row["relative_difference"]),
                fmt4(row["cost_difference"]),
                fmt4(row["bootstrap"]["ci_low"]),
                fmt4(row["bootstrap"]["ci_high"]),
                "yes" if row["bootstrap"]["includes_zero"] else "no",
            )
        )
    require(ci_matches == expected_ci, errors, f"Table 2 intervals do not match.\nParsed: {ci_matches}\nExpected: {expected_ci}")

    point_matches = re.findall(
        r"\$r=(\d+)\$,\s*(-?\d+\.\d+)\s+versus\s+(-?\d+\.\d+)",
        test,
    )
    effect_matches = re.findall(
        r"\$r=(\d+)\$,\s*\$\[(-?\d+\.\d+),\\ (-?\d+\.\d+)\]\$,\s*includes zero:\s*(yes|no)",
        test,
    )
    expected_points = []
    expected_effects = []
    for ratio in RATIOS:
        effect = results["by_ratio"][str(ratio)]["threshold_effect"]
        expected_points.append(
            (
                str(ratio),
                fmt4(effect["test_cost_at_0.50"]),
                fmt4(effect["test_cost_at_tuned_threshold"]),
            )
        )
        boot = effect["bootstrap"]
        expected_effects.append(
            (
                str(ratio),
                fmt4(boot["ci_low"]),
                fmt4(boot["ci_high"]),
                "yes" if boot["includes_zero"] else "no",
            )
        )
    require(point_matches == expected_points, errors, f"Threshold-effect costs do not match.\nParsed: {point_matches}\nExpected: {expected_points}")
    require(effect_matches == expected_effects, errors, f"Threshold-effect intervals do not match.\nParsed: {effect_matches}\nExpected: {expected_effects}")

    f1_line = re.search(
        r"logistic regression\s+(-?\d+\.\d+);\s*random forest\s+(-?\d+\.\d+);\s*"
        r"histogram GB\s+(-?\d+\.\d+);\s*XGBoost\s+(-?\d+\.\d+)",
        validation,
    )
    f1_values = results["validation_f1_at_0.50"]
    expected_f1 = (
        fmt4(f1_values["LogisticRegression"]),
        fmt4(f1_values["RandomForest"]),
        fmt4(f1_values["HistGradientBoosting"]),
        fmt4(f1_values["XGBoost"]),
    )
    require(f1_line is not None and f1_line.groups() == expected_f1, errors, f"Validation F1 at 0.50 does not match. Parsed: {None if f1_line is None else f1_line.groups()} Expected: {expected_f1}")
    if results["prediction_first"]["tie_break_used"]:
        require("cost tie-break was used" in validation, errors, "Manuscript omits that the cost tie-break was used.")
    else:
        require("cost tie-break was not used" in validation, errors, "Manuscript omits that the cost tie-break was not used.")

    rate = f"{100 * results['dataset']['default_rate']:.2f}\\%"
    require(rate in tex, errors, f"Manuscript does not contain the default rate {rate}.")
    defaults = f"{results['dataset']['n_defaults']:,}"
    require(defaults in tex, errors, f"Manuscript does not contain the default count {defaults}.")
    for phrase in (
        "30,000",
        "18,000",
        "6,000",
        "random forest (200 trees, depth 10, minimum leaf size 5)",
        "histogram gradient boosting (200 iterations, depth 6, learning rate 0.1)",
        "XGBoost (200 trees, depth 5, learning rate 0.1)",
        "logistic regression uses $C = 1$",
        "0.01 to 0.99 in steps of 0.01",
        "The split and the bootstrap each use seed 42.",
        "1,000",
        "SEX",
    ):
        require(phrase in tex, errors, f"Manuscript is missing the phrase: {phrase}")

    for ratio in RATIOS:
        row = results["by_ratio"][str(ratio)]
        if row["bootstrap"]["includes_zero"]:
            require(
                "not statistically conclusive" in tex,
                errors,
                "A confidence interval includes zero, but the manuscript never says the difference is not statistically conclusive.",
            )
        # Do not allow a blanket significance claim. Specific interval language is checked above.
        if "statistically significant" in tex.lower() or re.search(r"\bp\s*<\s*0\.", tex):
            errors.append("Manuscript uses a significance claim or a p-value. Report the interval instead.")

    summary = results["repeat_summary"]
    n_seeds = summary["n_seeds"]
    require("seeds 42 through 61" in tex, errors, "Manuscript does not name seeds 42 through 61.")
    require(f"{n_seeds} splits" in tex, errors, "Manuscript does not state the repeated-split count.")
    require("not a confidence interval" in tex, errors, "Manuscript does not say the split range is not a confidence interval.")
    counts = summary["prediction_first_model_counts"]
    model_phrase = (
        f"histogram gradient boosting on {counts['HistGradientBoosting']} splits, "
        f"XGBoost on {counts['XGBoost']}, and the random forest on {counts['RandomForest']}"
    )
    require(model_phrase in tex, errors, f"Prediction-first model counts do not match: {model_phrase}")
    require("No split used the cost tie-break." in tex, errors, "Manuscript does not state that no split used the cost tie-break.")
    n_ties = sum(1 for item in results["repeats"] if item["tie_break_used"])
    require(n_ties == 0, errors, f"A repeated split used the cost tie-break ({n_ties}), but the manuscript says none did.")

    def triple(band: dict) -> str:
        return f"{fmt4(band['min'])}, {fmt4(band['median'])}, {fmt4(band['max'])}"

    cost_a = []
    cost_b = []
    cost_half = []
    for ratio in RATIOS:
        band = summary["by_ratio"][str(ratio)]
        cost_a.append(f"$r={ratio}$, {triple(band['test_cost_a'])}")
        cost_b.append(f"$r={ratio}$, {triple(band['test_cost_b'])}")
        cost_half.append(f"$r={ratio}$, {triple(band['test_cost_at_0.50'])}")
        require(
            f"{fmt2(band['threshold_a']['min'])} to {fmt2(band['threshold_a']['max'])}" in tex,
            errors,
            f"Prediction-first threshold range for r={ratio} is missing.",
        )
    require(
        "prediction-first test cost min, median, and max were " + "; ".join(cost_a) in test,
        errors,
        "Table 2 is missing the prediction-first cost range.",
    )
    require(
        "Cost at threshold 0.50 min, median, and max were " + "; ".join(cost_half) in test,
        errors,
        "Table 2 is missing the threshold-0.50 cost range.",
    )
    require(
        "Decision-first test cost min, median, and max were " + "; ".join(cost_b) in test,
        errors,
        "Table 2 is missing the decision-first cost range.",
    )
    tuned_counts = [summary["by_ratio"][str(ratio)]["n_seeds_tuned_cost_below_0.50"] for ratio in RATIOS]
    decision_counts = [summary["by_ratio"][str(ratio)]["n_seeds_decision_first_lower_cost"] for ratio in RATIOS]
    require(
        tuned_counts[1] == tuned_counts[2] == tuned_counts[3],
        errors,
        "The compact tuned-cost sentence assumes equal counts at r=5, 10, and 20.",
    )
    require(
        (
            f"Tuned cost was lower on {tuned_counts[0]} of {n_seeds} splits at $r=2$ "
            f"and on {tuned_counts[1]} of {n_seeds} splits at $r=5$, $r=10$, and $r=20$."
        )
        in test,
        errors,
        "Table 2 tuned-cost counts do not match.",
    )
    require(
        (
            "Decision-first test cost was strictly lower on "
            f"{decision_counts[0]}, {decision_counts[1]}, {decision_counts[2]}, and {decision_counts[3]} "
            f"of {n_seeds} splits."
        )
        in test,
        errors,
        "Table 2 decision-first counts do not match.",
    )
    forest_counts = [
        sum(1 for item in results["repeats"] if item["by_ratio"][str(ratio)]["b_model"] == "RandomForest")
        for ratio in RATIOS
    ]
    require(
        (
            f"random forest on {forest_counts[0]}, {forest_counts[1]}, {forest_counts[2]}, "
            f"and {forest_counts[3]} of the {n_seeds} splits"
        )
        in tex,
        errors,
        "Decision-first random-forest counts do not match.",
    )
    n_edge = sum(1 for item in results["repeats"] if abs(item["by_ratio"]["20"]["a_threshold"] - 0.01) < 1e-9)
    require(f"{n_edge} of the {n_seeds} splits" in tex, errors, "Grid-edge count for r=20 does not match.")

    if errors:
        print("Manuscript check failed:", flush=True)
        for error in errors:
            print(f"- {error}", flush=True)
        raise SystemExit(1)
    print("Manuscript numbers match results.json.", flush=True)


def print_summary(results: dict) -> None:
    print("Default rate", f"{100 * results['dataset']['default_rate']:.2f}%", "n_defaults", results["dataset"]["n_defaults"])
    print("Validation F1 at 0.50")
    for name in MODEL_ORDER:
        print(f"  {DISPLAY[name]}: {fmt4(results['validation_f1_at_0.50'][name])}")
    print(
        "Prediction-first",
        results["prediction_first"]["display_name"],
        "tie_break_used",
        results["prediction_first"]["tie_break_used"],
    )
    for ratio in RATIOS:
        row = results["by_ratio"][str(ratio)]
        rule_a = row["strategy_a"]
        rule_b = row["strategy_b"]
        boot = row["bootstrap"]
        effect = row["threshold_effect"]
        print(
            f"r={ratio} A {rule_a['display_name']} t={fmt2(rule_a['threshold'])} "
            f"val={fmt4(rule_a['validation_cost'])}/{fmt4(rule_a['validation_f1'])} "
            f"test={fmt4(rule_a['test_cost'])} | "
            f"B {rule_b['display_name']} t={fmt2(rule_b['threshold'])} "
            f"val={fmt4(rule_b['validation_cost'])}/{fmt4(rule_b['validation_f1'])} "
            f"test={fmt4(rule_b['test_cost'])} | "
            f"rel={fmt4(row['relative_difference'])} diff={fmt4(row['cost_difference'])} "
            f"CI=[{fmt4(boot['ci_low'])}, {fmt4(boot['ci_high'])}] zero={boot['includes_zero']}"
        )
        print(
            f"  0.50 vs tuned {fmt4(effect['test_cost_at_0.50'])} vs {fmt4(effect['test_cost_at_tuned_threshold'])} "
            f"CI=[{fmt4(effect['bootstrap']['ci_low'])}, {fmt4(effect['bootstrap']['ci_high'])}] "
            f"zero={effect['bootstrap']['includes_zero']}"
        )
        band = results["repeat_summary"]["by_ratio"][str(ratio)]
        print(
            f"  seeds n={results['repeat_summary']['n_seeds']} "
            f"A t [{fmt2(band['threshold_a']['min'])}, {fmt2(band['threshold_a']['median'])}, {fmt2(band['threshold_a']['max'])}] "
            f"B t [{fmt2(band['threshold_b']['min'])}, {fmt2(band['threshold_b']['median'])}, {fmt2(band['threshold_b']['max'])}] "
            f"A cost [{fmt4(band['test_cost_a']['min'])}, {fmt4(band['test_cost_a']['median'])}, {fmt4(band['test_cost_a']['max'])}] "
            f"B cost [{fmt4(band['test_cost_b']['min'])}, {fmt4(band['test_cost_b']['median'])}, {fmt4(band['test_cost_b']['max'])}] "
            f"0.50 [{fmt4(band['test_cost_at_0.50']['min'])}, {fmt4(band['test_cost_at_0.50']['median'])}, {fmt4(band['test_cost_at_0.50']['max'])}] "
            f"tuned<0.50 {band['n_seeds_tuned_cost_below_0.50']}/{results['repeat_summary']['n_seeds']} "
            f"B<A {band['n_seeds_decision_first_lower_cost']}/{results['repeat_summary']['n_seeds']}"
        )
    print("prediction-first counts", results["repeat_summary"]["prediction_first_model_counts"])
    print(f"Elapsed seconds: {results['elapsed_seconds']:.1f}")


def main() -> None:
    results = run_experiment()
    (ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    save_figure_1(ROOT / "Figure_1.pdf", results)
    save_figure_2(ROOT / "Figure_2.pdf", results)
    assert_vector_pdf(ROOT / "Figure_1.pdf")
    assert_vector_pdf(ROOT / "Figure_2.pdf")
    print_summary(results)
    check_manuscript(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
