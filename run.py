#!/usr/bin/env python3
"""Reproducible experiment for the short communication.

Knowledge-Based Cost-Aware Decision Support for Credit Classification:
Separating Model Selection from Decision Thresholds.

The reported tables are the files in results/. This script regenerates them
from the UCI Default of Credit Card Clients workbook.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xlrd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost
    from xgboost import XGBClassifier
except Exception as exc:
    raise SystemExit("xgboost is required. Install requirements.txt first.") from exc

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

COST_RATIOS = (2, 5, 10, 20)
PRIMARY_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)
FINE_GRID_R20 = np.round(np.arange(0.001, 0.010, 0.001), 3)
PRIMARY_SEED = 42
REPEAT_SEEDS = tuple(range(42, 62))
MODEL_SEED = 42
N_BOOT = 1000
MODEL_ORDER = ("logistic regression", "random forest", "histogram GB", "XGBoost")

# Okabe-Ito, colorblind-safe. Figure captions live in the manuscript.
BLUE = "#0072B2"
VERMILION = "#D55E00"
GREEN = "#009E73"
GRAY = "#666666"
RATIO_STYLE = {
    2: ("#0072B2", "-"),
    5: ("#E69F00", "--"),
    10: ("#009E73", "-."),
    20: ("#D55E00", ":"),
}


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


def load_xls(path: Path) -> pd.DataFrame:
    """Read the UCI workbook with xlrd. pandas 3 rejects xlrd 1.2."""
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
    if frame.isna().any().any():
        raise SystemExit("The UCI file has missing values; this script does not impute.")
    return frame


def load_data(path: Path):
    if path.suffix.lower() == ".xls":
        raw = load_xls(path)
    elif path.suffix.lower() in {".xlsx"}:
        raw = pd.read_excel(path, header=1)
        if not any("default" in str(column).lower() for column in raw.columns):
            raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path)

    raw.columns = [str(column).strip() for column in raw.columns]
    target_candidates = [
        column
        for column in raw.columns
        if column.lower() in {"default payment next month", "default.payment.next.month", "y"}
    ]
    if not target_candidates:
        target_candidates = [column for column in raw.columns if "default" in column.lower()]
    if len(target_candidates) != 1:
        raise ValueError(f"Could not uniquely identify target. Columns: {raw.columns.tolist()}")
    target = target_candidates[0]
    drop = [column for column in raw.columns if column.lower() == "id"]
    sex = [column for column in raw.columns if column.lower() == "sex"]
    features = raw.drop(columns=[target] + drop + sex).copy()
    labels = pd.to_numeric(raw[target], errors="raise").astype(int)
    for column in features.columns:
        features[column] = pd.to_numeric(features[column], errors="raise")
    if len(raw) != 30000:
        print(f"WARNING: expected 30,000 rows, found {len(raw):,}.")
    if features.shape[1] != 22:
        raise SystemExit(f"Expected 22 predictors after dropping ID and SEX, found {features.shape[1]}.")
    return features, labels, raw, target


def split_data(features, labels, seed):
    # First hold out 20% test = 6,000. Then 25% of the remaining 24,000 = 6,000 validation.
    x_tv, x_test, y_tv, y_test = train_test_split(
        features, labels, test_size=0.20, stratify=labels, random_state=seed
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=seed
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def make_models():
    # early_stopping is the library default. On this training size that default
    # stops early. subsample and colsample_bytree are also library defaults.
    # n_jobs=-1 matches the run that produced results/.
    return {
        "logistic regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=1.0, penalty="l2", max_iter=5000)),
            ]
        ),
        "random forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=MODEL_SEED,
            n_jobs=-1,
        ),
        "histogram GB": HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=MODEL_SEED,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=MODEL_SEED,
            n_jobs=-1,
        ),
    }


def predict_at(prob, threshold):
    return (np.asarray(prob) >= float(threshold)).astype(int)


def counts(labels, prob, threshold):
    predicted = predict_at(prob, threshold)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return int(fp), int(fn), int(tp), int(tn)


def norm_cost(labels, prob, threshold, ratio):
    fp, fn, _, _ = counts(labels, prob, threshold)
    return (fp + ratio * fn) / len(labels)


def f1_at(labels, prob, threshold):
    return f1_score(labels, predict_at(prob, threshold), pos_label=1, zero_division=0)


def best_threshold(labels, prob, ratio, grid=PRIMARY_GRID):
    rows = [
        {"threshold": float(threshold), "cost": norm_cost(labels, prob, threshold, ratio), "f1": f1_at(labels, prob, threshold)}
        for threshold in grid
    ]
    rows.sort(key=lambda row: (row["cost"], -row["f1"], row["threshold"]))
    return rows[0], rows


def fit_one_split(features, labels, split_seed):
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(features, labels, split_seed)
    fitted = {}
    p_val = {}
    p_test = {}
    base = []
    for name in MODEL_ORDER:
        model = make_models()[name]
        model.fit(x_train, y_train)
        fitted[name] = model
        p_val[name] = model.predict_proba(x_val)[:, 1]
        p_test[name] = model.predict_proba(x_test)[:, 1]
        base.append(
            {
                "model": name,
                "val_f1_050": f1_at(y_val, p_val[name], 0.50),
                "val_brier": brier_score_loss(y_val, p_val[name]),
            }
        )

    max_f1 = max(row["val_f1_050"] for row in base)
    tied = [row["model"] for row in base if np.isclose(row["val_f1_050"], max_f1, atol=1e-15)]
    results = []
    curves = {}
    for ratio in COST_RATIOS:
        if len(tied) == 1:
            a_model = tied[0]
        else:
            a_model = sorted(tied, key=lambda name: (norm_cost(y_val, p_val[name], 0.50, ratio), name))[0]
        a_best, a_curve = best_threshold(y_val, p_val[a_model], ratio)
        curves[(ratio, a_model)] = a_curve
        candidates = []
        for name in MODEL_ORDER:
            b_best, _ = best_threshold(y_val, p_val[name], ratio)
            candidates.append({"model": name, **b_best})
        candidates.sort(key=lambda row: (row["cost"], -row["f1"], row["model"], row["threshold"]))
        chosen_b = candidates[0]
        for strategy, name, threshold in (
            ("A", a_model, a_best["threshold"]),
            ("B", chosen_b["model"], chosen_b["threshold"]),
        ):
            results.append(
                {
                    "split_seed": split_seed,
                    "r": ratio,
                    "strategy": strategy,
                    "model": name,
                    "threshold": threshold,
                    "val_cost": norm_cost(y_val, p_val[name], threshold, ratio),
                    "val_f1": f1_at(y_val, p_val[name], threshold),
                    "test_cost": norm_cost(y_test, p_test[name], threshold, ratio),
                    "test_f1": f1_at(y_test, p_test[name], threshold),
                }
            )
        results.append(
            {
                "split_seed": split_seed,
                "r": ratio,
                "strategy": "A@0.50",
                "model": a_model,
                "threshold": 0.50,
                "val_cost": norm_cost(y_val, p_val[a_model], 0.50, ratio),
                "val_f1": f1_at(y_val, p_val[a_model], 0.50),
                "test_cost": norm_cost(y_test, p_test[a_model], 0.50, ratio),
                "test_f1": f1_at(y_test, p_test[a_model], 0.50),
            }
        )
    return {
        "base": pd.DataFrame(base),
        "results": pd.DataFrame(results),
        "curves": curves,
        "y_val": np.asarray(y_val),
        "y_test": np.asarray(y_test),
        "p_val": p_val,
        "p_test": p_test,
    }


def paired_bootstrap_cost(labels, p1, t1, p2, t2, ratio, seed=42, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    n = len(labels)
    diffs = np.empty(n_boot)
    labels = np.asarray(labels)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    for draw in range(n_boot):
        index = rng.integers(0, n, n)
        diffs[draw] = norm_cost(labels[index], p1[index], t1, ratio) - norm_cost(labels[index], p2[index], t2, ratio)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def bootstrap_mean(values, seed=42, n_boot=N_BOOT):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for draw in range(n_boot):
        means[draw] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def primary_bootstrap(primary):
    res = primary["results"]
    labels = primary["y_test"]
    p_test = primary["p_test"]
    rows = []
    for ratio in COST_RATIOS:
        a = res[(res.r == ratio) & (res.strategy == "A")].iloc[0]
        b = res[(res.r == ratio) & (res.strategy == "B")].iloc[0]
        half = res[(res.r == ratio) & (res.strategy == "A@0.50")].iloc[0]
        low, high = paired_bootstrap_cost(labels, p_test[a.model], a.threshold, p_test[b.model], b.threshold, ratio)
        t_low, t_high = paired_bootstrap_cost(labels, p_test[a.model], 0.50, p_test[a.model], a.threshold, ratio)
        rows.append(
            {
                "r": ratio,
                "A_cost": a.test_cost,
                "B_cost": b.test_cost,
                "A_minus_B": a.test_cost - b.test_cost,
                "relative_A_minus_B_over_A": (a.test_cost - b.test_cost) / a.test_cost,
                "strategy_ci_low": low,
                "strategy_ci_high": high,
                "A_050_cost": half.test_cost,
                "threshold_gain_050_minus_tuned": half.test_cost - a.test_cost,
                "threshold_ci_low": t_low,
                "threshold_ci_high": t_high,
            }
        )
    return pd.DataFrame(rows)


def r20_sensitivity(primary):
    res = primary["results"]
    a = res[(res.r == 20) & (res.strategy == "A")].iloc[0]
    fine_best, _ = best_threshold(primary["y_val"], primary["p_val"][a.model], 20, FINE_GRID_R20)
    return pd.DataFrame(
        [
            {
                "model": a.model,
                "primary_threshold": a.threshold,
                "primary_val_cost": a.val_cost,
                "primary_test_cost": a.test_cost,
                "fine_threshold": fine_best["threshold"],
                "fine_val_cost": fine_best["cost"],
                "fine_test_cost": norm_cost(primary["y_test"], primary["p_test"][a.model], fine_best["threshold"], 20),
            }
        ]
    )


def stability_table(all_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ratio in COST_RATIOS:
        a = all_results[(all_results.r == ratio) & (all_results.strategy == "A")].sort_values("split_seed")
        b = all_results[(all_results.r == ratio) & (all_results.strategy == "B")].sort_values("split_seed")
        half = all_results[(all_results.r == ratio) & (all_results.strategy == "A@0.50")].sort_values("split_seed")
        threshold_gain = half.test_cost.to_numpy() - a.test_cost.to_numpy()
        strategy_gap = a.test_cost.to_numpy() - b.test_cost.to_numpy()
        gain_ci = bootstrap_mean(threshold_gain)
        gap_ci = bootstrap_mean(strategy_gap)
        rows.append(
            {
                "r": ratio,
                "A_cost_min": a.test_cost.min(),
                "A_cost_median": a.test_cost.median(),
                "A_cost_max": a.test_cost.max(),
                "A050_cost_min": half.test_cost.min(),
                "A050_cost_median": half.test_cost.median(),
                "A050_cost_max": half.test_cost.max(),
                "B_cost_min": b.test_cost.min(),
                "B_cost_median": b.test_cost.median(),
                "B_cost_max": b.test_cost.max(),
                "tuned_beats_050_n": int((threshold_gain > 0).sum()),
                "B_beats_A_n": int((strategy_gap > 0).sum()),
                "A_beats_B_n": int((strategy_gap < 0).sum()),
                "ties_n": int(np.isclose(strategy_gap, 0, atol=1e-15).sum()),
                "mean_threshold_gain": threshold_gain.mean(),
                "threshold_gain_ci_low": gain_ci[0],
                "threshold_gain_ci_high": gain_ci[1],
                "mean_strategy_gap": strategy_gap.mean(),
                "strategy_gap_ci_low": gap_ci[0],
                "strategy_gap_ci_high": gap_ci[1],
            }
        )
    return pd.DataFrame(rows)


def curves_frame(primary) -> pd.DataFrame:
    rows = []
    res = primary["results"]
    for ratio in COST_RATIOS:
        a = res[(res.r == ratio) & (res.strategy == "A")].iloc[0]
        for point in primary["curves"][(ratio, a.model)]:
            rows.append(
                {
                    "r": ratio,
                    "model": a.model,
                    "threshold": point["threshold"],
                    "val_cost": point["cost"],
                }
            )
    return pd.DataFrame(rows)


def _paper_rc(size):
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": size,
            "axes.labelsize": size,
            "xtick.labelsize": size - 1,
            "ytick.labelsize": size - 1,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
        }
    )


def save_figure_1(path: Path, curves: pd.DataFrame, primary_results: pd.DataFrame) -> None:
    _paper_rc(7.5)
    fig, axes = plt.subplots(2, 2, figsize=(9 / 2.54, 7.6 / 2.54), sharex=True)
    panel = {2: "a", 5: "b", 10: "c", 20: "d"}
    for axis, ratio in zip(axes.ravel(), COST_RATIOS):
        curve = curves[curves.r == ratio].sort_values("threshold")
        color, style = RATIO_STYLE[ratio]
        chosen = float(primary_results[(primary_results.r == ratio) & (primary_results.strategy == "A")].threshold.iloc[0])
        axis.plot(curve.threshold, curve.val_cost, color=color, linestyle=style, linewidth=1.15)
        chosen_cost = float(curve.loc[np.isclose(curve.threshold, chosen), "val_cost"].iloc[0])
        axis.plot([chosen], [chosen_cost], marker="o", color=color, markersize=4.8, zorder=4)
        y_span = float(curve.val_cost.max() - curve.val_cost.min())
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
    _save_figure(fig, path)
    plt.close(fig)


def save_figure_2(path: Path, primary_results: pd.DataFrame, all_results: pd.DataFrame) -> None:
    _paper_rc(8)
    positions = np.arange(len(COST_RATIOS))

    def series(strategy, column):
        values = []
        low = []
        high = []
        for ratio in COST_RATIOS:
            seed42 = float(
                primary_results[(primary_results.r == ratio) & (primary_results.strategy == strategy)][column].iloc[0]
            )
            sample = all_results[(all_results.r == ratio) & (all_results.strategy == strategy)][column].to_numpy()
            values.append(seed42)
            low.append(seed42 - float(sample.min()))
            high.append(float(sample.max()) - seed42)
        return np.array(values), np.vstack([low, high])

    threshold_a, whisker_a = series("A", "threshold")
    threshold_b, whisker_b = series("B", "threshold")
    cost_a, cost_whisker_a = series("A", "test_cost")
    cost_b, cost_whisker_b = series("B", "test_cost")
    cost_half, cost_whisker_half = series("A@0.50", "test_cost")

    fig, axes = plt.subplots(2, 1, figsize=(9 / 2.54, 8.2 / 2.54), sharex=True)
    axes[0].errorbar(positions, threshold_a, yerr=whisker_a, color=BLUE, marker="o", linestyle="-", label="A prediction-first", linewidth=1.2, markersize=4.5, capsize=2, elinewidth=0.7)
    axes[0].errorbar(positions, threshold_b, yerr=whisker_b, color=VERMILION, marker="s", linestyle="--", label="B decision-first", linewidth=1.2, markersize=4.5, capsize=2, elinewidth=0.7)
    axes[0].axhline(0.50, color=GRAY, linestyle=":", linewidth=0.9, label="threshold 0.50")
    limits = list(threshold_a) + list(threshold_b) + [0.50]
    limits += [float(all_results[(all_results.r == ratio) & (all_results.strategy == "A")].threshold.min()) for ratio in COST_RATIOS]
    limits += [float(all_results[(all_results.r == ratio) & (all_results.strategy == "B")].threshold.max()) for ratio in COST_RATIOS]
    axes[0].set_ylim(min(limits) - 0.05, 0.64)
    axes[0].set_ylabel("Threshold")
    axes[0].text(0.02, 0.90, "(a)", transform=axes[0].transAxes, fontsize=8, fontweight="bold")
    axes[0].legend(frameon=False, loc="upper right", bbox_to_anchor=(0.99, 0.76), handlelength=1.5, borderaxespad=0.0)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.4)

    axes[1].errorbar(positions, cost_a, yerr=cost_whisker_a, color=BLUE, marker="o", linestyle="-", label="A prediction-first", linewidth=1.2, markersize=4.5, capsize=2, elinewidth=0.7)
    axes[1].errorbar(positions, cost_b, yerr=cost_whisker_b, color=VERMILION, marker="s", linestyle="--", label="B decision-first", linewidth=1.2, markersize=4.5, capsize=2, elinewidth=0.7)
    axes[1].errorbar(positions, cost_half, yerr=cost_whisker_half, color=GREEN, marker="x", linestyle=":", label="A at 0.50", linewidth=1.2, markersize=5, capsize=2, elinewidth=0.7)
    axes[1].set_ylabel("Test cost")
    axes[1].set_xlabel("Cost ratio r")
    axes[1].text(0.02, 0.90, "(b)", transform=axes[1].transAxes, fontsize=8, fontweight="bold")
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.14, 1.0), handlelength=1.6, borderaxespad=0.1)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.4)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([str(ratio) for ratio in COST_RATIOS])
    fig.tight_layout(h_pad=0.8)
    _save_figure(fig, path)
    plt.close(fig)


def _save_figure(fig, path: Path) -> None:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    else:
        fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.04)


def assert_vector_pdf(path: Path) -> None:
    data = path.read_bytes()
    if not data.startswith(b"%PDF"):
        raise SystemExit(f"{path.name} is not a PDF.")
    if b"/Subtype /Image" in data or b"/Subtype/Image" in data:
        raise SystemExit(f"{path.name} contains a raster image.")
    if b"/Font" not in data:
        raise SystemExit(f"{path.name} does not embed a font.")


def write_package_figures(curves, primary_results, all_results) -> None:
    save_figure_1(ROOT / "Figure_1.pdf", curves, primary_results)
    save_figure_2(ROOT / "Figure_2.pdf", primary_results, all_results)
    assert_vector_pdf(ROOT / "Figure_1.pdf")
    assert_vector_pdf(ROOT / "Figure_2.pdf")
    save_figure_1(Path("/tmp/kbs-fig1-1.png"), curves, primary_results)
    save_figure_2(Path("/tmp/kbs-fig2-1.png"), primary_results, all_results)


def write_preview_pngs(primary, all_results, outdir: Path) -> None:
    res = primary["results"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), sharex=True)
    for axis, ratio in zip(axes.flat, COST_RATIOS):
        a = res[(res.r == ratio) & (res.strategy == "A")].iloc[0]
        curve = primary["curves"][(ratio, a.model)]
        axis.plot([point["threshold"] for point in curve], [point["cost"] for point in curve])
        axis.axvline(0.50, linestyle=":")
        axis.scatter([a.threshold], [a.val_cost], zorder=3)
        axis.annotate(f"{a.threshold:.2f}", (a.threshold, a.val_cost), xytext=(5, 8), textcoords="offset points")
        axis.set_title(f"({chr(97 + list(COST_RATIOS).index(ratio))})  r = {ratio}")
        axis.set_ylabel("Val. cost")
        axis.set_xlabel("Threshold")
    fig.tight_layout()
    fig.savefig(outdir / "figure1_validation_cost_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    for strategy, marker, label in (("A", "o", "A prediction-first"), ("B", "s", "B decision-first")):
        means, low, high = [], [], []
        for ratio in COST_RATIOS:
            sample = all_results[(all_results.r == ratio) & (all_results.strategy == strategy)].threshold.to_numpy()
            seed42 = float(res[(res.r == ratio) & (res.strategy == strategy)].threshold.iloc[0])
            means.append(seed42)
            low.append(seed42 - sample.min())
            high.append(sample.max() - seed42)
        axes[0].errorbar(COST_RATIOS, means, yerr=[low, high], marker=marker, label=label, capsize=4)
    axes[0].axhline(0.50, linestyle=":", label="threshold 0.50")
    axes[0].set_ylabel("Threshold")
    axes[0].legend()
    for strategy, marker, label in (
        ("A", "o", "A prediction-first"),
        ("B", "s", "B decision-first"),
        ("A@0.50", "x", "A at 0.50"),
    ):
        values, low, high = [], [], []
        for ratio in COST_RATIOS:
            sample = all_results[(all_results.r == ratio) & (all_results.strategy == strategy)].test_cost.to_numpy()
            seed42 = float(res[(res.r == ratio) & (res.strategy == strategy)].test_cost.iloc[0])
            values.append(seed42)
            low.append(seed42 - sample.min())
            high.append(sample.max() - seed42)
        axes[1].errorbar(COST_RATIOS, values, yerr=[low, high], marker=marker, label=label, capsize=4)
    axes[1].set_ylabel("Test cost")
    axes[1].set_xlabel("Cost ratio r")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(outdir / "figure2_threshold_and_test_cost.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_tables(outdir: Path):
    base = pd.read_csv(outdir / "primary_model_validation_metrics.csv")
    primary = pd.read_csv(outdir / "primary_strategy_results.csv")
    boot = pd.read_csv(outdir / "primary_bootstrap_contrasts.csv")
    sens = pd.read_csv(outdir / "r20_fine_grid_sensitivity.csv")
    stability = pd.read_csv(outdir / "20_split_stability_summary.csv")
    all_results = pd.read_csv(outdir / "all_20_split_results.csv")
    return base, primary, boot, sens, stability, all_results


def manuscript_phrases(base, primary, boot, sens, stability, all_results):
    phrases = []
    by_model = {row.model: row for _, row in base.iterrows()}
    phrases.append(fmt4(by_model["XGBoost"].val_f1_050))
    phrases.append(fmt4(by_model["histogram GB"].val_f1_050))
    phrases.append(f"{fmt4(by_model['histogram GB'].val_brier)} for histogram gradient boosting")
    phrases.append(f"{fmt4(by_model['logistic regression'].val_brier)} for logistic regression")
    a_rows = primary[primary.strategy == "A"].sort_values("r")
    thresholds = [fmt2(value) for value in a_rows.threshold]
    phrases.append(f"{thresholds[0]}, {thresholds[1]}, {thresholds[2]}, and {thresholds[3]}")
    for _, row in primary[primary.strategy.isin(["A", "B"])].iterrows():
        phrases.append(fmt4(row.val_cost))
        phrases.append(fmt4(row.val_f1))
        phrases.append(fmt4(row.test_cost))
    for _, row in boot.iterrows():
        phrases.append(fmt4(row.A_minus_B))
        phrases.append(fmt4(row.relative_A_minus_B_over_A))
        phrases.append(fmt4(row.strategy_ci_low))
        phrases.append(fmt4(row.strategy_ci_high))
        phrases.append(fmt4(row.A_050_cost))
        phrases.append(fmt4(row.threshold_ci_low))
        phrases.append(fmt4(row.threshold_ci_high))
    sens_row = sens.iloc[0]
    phrases.append(fmt4(sens_row.fine_val_cost))
    phrases.append(fmt4(sens_row.fine_test_cost))
    phrases.append(fmt4(sens_row.primary_val_cost))
    phrases.append(fmt4(sens_row.primary_test_cost))
    a = all_results[all_results.strategy == "A"]
    edge = int(np.isclose(a[a.r == 20].threshold, 0.01).sum())
    phrases.append(f"{edge} of the 20 splits")
    counts = a.groupby("split_seed").model.first().value_counts()
    phrases.append(f"XGBoost on {int(counts.get('XGBoost', 0))} splits")
    phrases.append(f"histogram gradient boosting on {int(counts.get('histogram GB', 0))}")
    phrases.append(f"random forest on {int(counts.get('random forest', 0))}")
    gains = [fmt4(value) for value in stability.mean_threshold_gain]
    phrases.append(f"{gains[0]}, {gains[1]}, {gains[2]}, and {gains[3]}")
    for value in stability.mean_strategy_gap:
        phrases.append(fmt4(value))
    for _, row in stability.iterrows():
        for column in (
            "A_cost_min",
            "A_cost_median",
            "A_cost_max",
            "A050_cost_min",
            "A050_cost_median",
            "A050_cost_max",
            "B_cost_min",
            "B_cost_median",
            "B_cost_max",
            "threshold_gain_ci_low",
            "threshold_gain_ci_high",
            "strategy_gap_ci_low",
            "strategy_gap_ci_high",
        ):
            phrases.append(fmt4(row[column]))
    b_counts = []
    for ratio in COST_RATIOS:
        subset = all_results[(all_results.strategy == "B") & (all_results.r == ratio)]
        b_counts.append(str(int((subset.model == "random forest").sum())))
    phrases.append(", ".join(b_counts[:3]) + ", and " + b_counts[3])
    return phrases


def check_manuscript(outdir: Path) -> None:
    tables = load_tables(outdir)
    text = (ROOT / "manuscript.tex").read_text()
    missing = [phrase for phrase in manuscript_phrases(*tables) if phrase not in text]
    stale = [phrase for phrase in ("0.4682", "0.4598", "0.1368", "0.7522", "0.5762", "9 of the 20", "6 of the 20") if phrase in text]
    if missing or stale:
        print("Manuscript check failed.")
        for phrase in missing:
            print(f"  missing: {phrase}")
        for phrase in stale:
            print(f"  stale: {phrase}")
        raise SystemExit(1)
    abstract = text.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]
    words = abstract.replace(r"\%", "%").split()
    if len(words) > 250:
        raise SystemExit(f"Abstract has {len(words)} words; the limit is 250.")
    print(f"Manuscript numbers match results/. Abstract words: {len(words)}.")


def frames_close(left: pd.DataFrame, right: pd.DataFrame, keys, columns) -> bool:
    merged = left.merge(right, on=keys, suffixes=("_fit", "_csv"))
    if len(merged) != len(left) or len(merged) != len(right):
        return False
    for column in columns:
        if not np.allclose(merged[f"{column}_fit"], merged[f"{column}_csv"], rtol=0, atol=1e-9):
            return False
    return True


def metadata() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "primary_seed": PRIMARY_SEED,
        "model_seed": MODEL_SEED,
        "repeat_seeds": list(REPEAT_SEEDS),
        "bootstrap_resamples": N_BOOT,
        "primary_grid": [float(PRIMARY_GRID[0]), float(PRIMARY_GRID[-1]), 0.01],
        "r20_fine_grid": [float(FINE_GRID_R20[0]), float(FINE_GRID_R20[-1]), 0.001],
        "cost_ratios": list(COST_RATIOS),
    }


def run_experiment(features, labels, outdir: Path):
    print(
        f"Rows={len(labels):,}; defaults={int(labels.sum()):,}; rate={labels.mean():.4%}; predictors={features.shape[1]}",
        flush=True,
    )
    primary = fit_one_split(features, labels, PRIMARY_SEED)
    primary["base"].to_csv(outdir / "primary_model_validation_metrics.csv", index=False)
    primary["results"].to_csv(outdir / "primary_strategy_results.csv", index=False)
    boot = primary_bootstrap(primary)
    boot.to_csv(outdir / "primary_bootstrap_contrasts.csv", index=False)
    sens = r20_sensitivity(primary)
    sens.to_csv(outdir / "r20_fine_grid_sensitivity.csv", index=False)
    frames = []
    for seed in REPEAT_SEEDS:
        print(f"Running split seed {seed}...", flush=True)
        frames.append(fit_one_split(features, labels, seed)["results"])
    all_results = pd.concat(frames, ignore_index=True)
    all_results.to_csv(outdir / "all_20_split_results.csv", index=False)
    stability = stability_table(all_results)
    stability.to_csv(outdir / "20_split_stability_summary.csv", index=False)
    table1 = primary["results"][primary["results"].strategy.isin(["A", "B"])][
        ["r", "strategy", "model", "threshold", "val_cost", "val_f1"]
    ]
    table2 = primary["results"][primary["results"].strategy.isin(["A", "B"])][
        ["r", "strategy", "model", "threshold", "test_cost"]
    ]
    table1.to_csv(outdir / "table1_validation_choices.csv", index=False)
    table2.to_csv(outdir / "table2_test_costs.csv", index=False)
    curves = curves_frame(primary)
    curves.to_csv(outdir / "validation_cost_curves.csv", index=False)
    write_preview_pngs(primary, all_results, outdir)
    write_package_figures(curves, primary["results"], all_results)
    (outdir / "run_metadata.json").write_text(json.dumps(metadata(), indent=2))
    print(f"Saved all outputs to: {outdir.resolve()}", flush=True)


def figures_from_locked_run(features, labels, outdir: Path) -> None:
    """Refit seed 42, require it to match the locked CSVs, then redraw the figures."""
    print("Refitting seed 42 to redraw the vector figures.", flush=True)
    primary = fit_one_split(features, labels, PRIMARY_SEED)
    locked_base = pd.read_csv(outdir / "primary_model_validation_metrics.csv")
    locked_results = pd.read_csv(outdir / "primary_strategy_results.csv")
    base_ok = frames_close(primary["base"], locked_base, ["model"], ["val_f1_050", "val_brier"])
    results_ok = frames_close(
        primary["results"],
        locked_results,
        ["r", "strategy"],
        ["threshold", "val_cost", "val_f1", "test_cost", "test_f1"],
    )
    if not (base_ok and results_ok):
        raise SystemExit("Seed-42 refit does not match the locked results/. Figures were not replaced.")
    curves = curves_frame(primary)
    curves.to_csv(outdir / "validation_cost_curves.csv", index=False)
    all_results = pd.read_csv(outdir / "all_20_split_results.csv")
    write_preview_pngs(primary, all_results, outdir)
    write_package_figures(curves, locked_results, all_results)
    print("Vector figures match the locked seed-42 results.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="UCI xls, xlsx, or csv. Default: download the UCI workbook.")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--check-only", action="store_true", help="Check manuscript.tex against results/ and exit.")
    parser.add_argument("--figures-only", action="store_true", help="Redraw figures from a seed-42 refit that must match results/.")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    if args.check_only:
        check_manuscript(outdir)
        return
    data_path = Path(args.data) if args.data else download_xls()
    features, labels, _, target = load_data(data_path)
    print(f"Target: {target}", flush=True)
    if args.figures_only:
        figures_from_locked_run(features, labels, outdir)
        check_manuscript(outdir)
        return
    run_experiment(features, labels, outdir)
    check_manuscript(outdir)


if __name__ == "__main__":
    main()
