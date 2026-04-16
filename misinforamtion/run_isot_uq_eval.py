#!/usr/bin/env python3
"""
Structured ISOT misinformation evaluation with LM-Polygraph.

This script:
1. loads ISOT (Fake.csv / True.csv),
2. frames the task as constrained generation (REAL vs FAKE),
3. runs black-box UQ methods with LM-Polygraph on the full sampled dataset,
4. saves clean tabular outputs,
5. generates interpretable plots and a short markdown report.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lm_polygraph.defaults.register_default_stat_calculators import (
    register_default_stat_calculators,
)
from lm_polygraph.estimators import (
    DegMat,
    Eccentricity,
    EigValLaplacian,
    KernelLanguageEntropy,
    LexicalSimilarity,
    LUQ,
    NumSemSets,
)
from lm_polygraph.utils.builder_enviroment_stat_calculator import (
    BuilderEnvironmentStatCalculator,
)
from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils.generation_parameters import GenerationParameters
from lm_polygraph.utils.manager import UEManager
from lm_polygraph.utils.model import BlackboxModel


PROMPT_TEMPLATE = """You are a careful misinformation detection assistant.
Read the following news item and classify it as REAL or FAKE.

Rules:
- Return exactly one final label.
- Do not explain your reasoning.
- Use one of these exact outputs:
FINAL ANSWER: REAL
FINAL ANSWER: FAKE

News item:
Title: {title}
Article:
{article}
"""

LABEL_PATTERN = re.compile(r"FINAL ANSWER:\s*(REAL|FAKE)\b", flags=re.IGNORECASE)
DEFAULT_ENV_PATH = SCRIPT_DIR / ".env"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"

METHOD_EXPLANATIONS = {
    "EigValLaplacian_NLI_score_entail": (
        "Measures how semantically diverse the sampled answers are. "
        "Higher scores mean the model's sampled answers disagree more in meaning."
    ),
    "EigValLaplacian_Jaccard_score": (
        "A lexical-overlap variant of EigValLaplacian. "
        "It measures disagreement through token overlap instead of NLI."
    ),
    "NumSemSets": (
        "Counts how many distinct semantic groups appear among sampled answers. "
        "Higher values mean the model explores more incompatible meanings."
    ),
    "LexicalSimilarity_rougeL": (
        "Uses surface-form similarity across sampled answers. "
        "Lower similarity between samples implies higher uncertainty."
    ),
    "LexicalSimilarity_rouge1": (
        "Surface-form overlap using ROUGE-1. "
        "Lower overlap between samples implies higher uncertainty."
    ),
    "DegMat_NLI_score_entail": (
        "Uses graph connectivity over semantic similarities. "
        "It reflects how tightly grouped the sampled answers are."
    ),
    "DegMat_Jaccard_score": (
        "A Jaccard-overlap variant of DegMat. "
        "It captures how tightly grouped sampled answers are at the lexical level."
    ),
    "Eccentricity_NLI_score_entail": (
        "Measures how spread out sampled answers are in a semantic graph. "
        "Higher values mean the generations occupy more distant semantic regions."
    ),
    "Eccentricity_Jaccard_score": (
        "A lexical-overlap variant of Eccentricity. "
        "Higher values mean sampled answers are dispersed in lexical space."
    ),
    "KernelLanguageEntropy": (
        "Computes entropy over a heat kernel built from semantic relations between samples. "
        "Higher values mean more uncertainty in the semantic output space."
    ),
    "LUQ": (
        "Long-text uncertainty score derived from entailment and contradiction logits "
        "between sampled responses. Higher values mean less semantic agreement."
    ),
}

VERBALIZED_EXPERIMENT_NOTE = (
    "Verbalized1S/2S are intentionally kept for a second experiment with a dedicated prompt "
    "that asks the model to report its confidence explicitly."
)


@dataclass
class RunArtifacts:
    run_dir: Path
    plots_dir: Path
    examples_compact_csv: Path
    examples_full_csv: Path
    uq_scores_long_csv: Path
    method_summary_csv: Path
    wrong_examples_csv: Path
    workbook_xlsx: Path
    report_md: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LM-Polygraph black-box UQ methods on ISOT with structured outputs."
    )
    parser.add_argument(
        "--fake-csv",
        type=Path,
        default=SCRIPT_DIR / "data" / "isot" / "Fake.csv",
        help="Path to ISOT Fake.csv",
    )
    parser.add_argument(
        "--true-csv",
        type=Path,
        default=SCRIPT_DIR / "data" / "isot" / "True.csv",
        help="Path to ISOT True.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base directory where run outputs will be saved",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional run name. If omitted, a timestamped name is generated",
    )
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=20,
        help="Number of examples per class to keep. Use -1 to keep the full dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for prompt evaluation",
    )
    parser.add_argument(
        "--max-title-chars",
        type=int,
        default=300,
        help="Maximum title length included in the prompt",
    )
    parser.add_argument(
        "--max-article-chars",
        type=int,
        default=3000,
        help="Maximum article length included in the prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Maximum generation length",
    )
    parser.add_argument(
        "--top-wrong-confident",
        type=int,
        default=10,
        help="Number of wrong but apparently confident examples to export",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help="Optional .env file containing OPENAI_API_KEY",
    )
    return parser.parse_args()


def load_env_file(env_file: Path) -> None:
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def validate_environment(env_file: Path) -> None:
    load_env_file(env_file)
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Put it in the environment or in misinforamtion/.env."
        )


def normalize_whitespace(text: str) -> str:
    text = "" if text is None else str(text)
    return re.sub(r"\s+", " ", text).strip()


def truncate_text(text: str, max_chars: int) -> str:
    text = normalize_whitespace(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def load_isot_dataset(
    fake_csv: Path,
    true_csv: Path,
    sample_per_class: int,
    seed: int,
) -> pd.DataFrame:
    fake_df = pd.read_csv(fake_csv).copy()
    true_df = pd.read_csv(true_csv).copy()

    fake_df["gold_label"] = "FAKE"
    fake_df["source_label"] = "Fake.csv"
    true_df["gold_label"] = "REAL"
    true_df["source_label"] = "True.csv"

    data = pd.concat([fake_df, true_df], ignore_index=True)
    required_columns = ["title", "text", "gold_label", "source_label"]
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise ValueError(f"ISOT files are missing required columns: {missing}")

    sampled_parts = []
    for _, group in data.groupby("gold_label", sort=True):
        if sample_per_class == -1:
            sampled_parts.append(group.copy())
        else:
            take_n = min(sample_per_class, len(group))
            sampled_parts.append(group.sample(n=take_n, random_state=seed))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sampled["example_id"] = np.arange(len(sampled))
    sampled["title"] = sampled["title"].map(normalize_whitespace)
    sampled["text"] = sampled["text"].map(normalize_whitespace)
    return sampled


def build_prompt(title: str, article: str, max_title_chars: int, max_article_chars: int) -> str:
    return PROMPT_TEMPLATE.format(
        title=truncate_text(title, max_title_chars) or "(empty title)",
        article=truncate_text(article, max_article_chars),
    )


def parse_prediction(raw_output: str) -> Optional[str]:
    if not raw_output:
        return None
    match = LABEL_PATTERN.search(raw_output)
    if match:
        return match.group(1).upper()
    upper = raw_output.upper()
    if "REAL" in upper and "FAKE" not in upper:
        return "REAL"
    if "FAKE" in upper and "REAL" not in upper:
        return "FAKE"
    return None


def build_model(max_new_tokens: int) -> BlackboxModel:
    generation_parameters = GenerationParameters(
        temperature=1.0,
        top_p=1.0,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        stop_strings=["\n\n"],
    )
    return BlackboxModel.from_openai(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model_path="gpt-4.1-mini",
        supports_logprobs=False,
        generation_parameters=generation_parameters,
    )


def build_estimators() -> List:
    return [
        EigValLaplacian(),
        EigValLaplacian(similarity_score="Jaccard_score"),
        NumSemSets(),
        LexicalSimilarity(metric="rougeL"),
        LexicalSimilarity(metric="rouge1"),
        DegMat(),
        DegMat(similarity_score="Jaccard_score"),
        Eccentricity(),
        Eccentricity(similarity_score="Jaccard_score"),
        KernelLanguageEntropy(),
        LUQ(),
    ]


def build_run_artifacts(base_dir: Path, run_name: str) -> RunArtifacts:
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"isot_gpt-4.1-mini_{timestamp}"
    run_dir = base_dir / run_name
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(
        run_dir=run_dir,
        plots_dir=plots_dir,
        examples_compact_csv=run_dir / "examples_compact.csv",
        examples_full_csv=run_dir / "examples_full.csv",
        uq_scores_long_csv=run_dir / "uq_scores_long.csv",
        method_summary_csv=run_dir / "method_summary.csv",
        wrong_examples_csv=run_dir / "wrong_but_confident.csv",
        workbook_xlsx=run_dir / "results.xlsx",
        report_md=run_dir / "report.md",
    )


def run_uq_pipeline(
    data_df: pd.DataFrame,
    model: BlackboxModel,
    estimators: List,
    batch_size: int,
    max_title_chars: int,
    max_article_chars: int,
) -> tuple[pd.DataFrame, Dict]:
    eval_df = data_df.copy()
    eval_df["prompt"] = [
        build_prompt(title, article, max_title_chars, max_article_chars)
        for title, article in zip(eval_df["title"], eval_df["text"])
    ]

    dataset = Dataset(
        x=eval_df["prompt"].tolist(),
        y=["" for _ in range(len(eval_df))],
        batch_size=batch_size,
    )

    manager = UEManager(
        data=dataset,
        model=model,
        estimators=estimators,
        builder_env_stat_calc=BuilderEnvironmentStatCalculator(model),
        available_stat_calculators=register_default_stat_calculators(
            model_type="Blackbox",
            blackbox_supports_logprobs=False,
        ),
        generation_metrics=[],
        ue_metrics=[],
        processors=[],
        ignore_exceptions=False,
        verbose=True,
        max_new_tokens=model.generation_parameters.max_new_tokens,
    )
    manager()

    eval_df["raw_output"] = manager.stats["greedy_texts"]
    eval_df["predicted_label"] = eval_df["raw_output"].map(parse_prediction)
    eval_df["parse_failed"] = eval_df["predicted_label"].isna().astype(int)
    eval_df["is_correct"] = (
        eval_df["predicted_label"].fillna("__MISSING__") == eval_df["gold_label"]
    ).astype(int)
    eval_df["is_incorrect"] = 1 - eval_df["is_correct"]

    for estimator in estimators:
        method_name = str(estimator)
        eval_df[method_name] = manager.estimations[(estimator.level, method_name)]

    return eval_df, manager.stats


def build_long_scores_df(eval_df: pd.DataFrame, uq_columns: List[str]) -> pd.DataFrame:
    long_df = eval_df.melt(
        id_vars=[
            "example_id",
            "gold_label",
            "predicted_label",
            "is_correct",
            "is_incorrect",
            "parse_failed",
            "title",
        ],
        value_vars=uq_columns,
        var_name="uq_method",
        value_name="uncertainty_score",
    )
    long_df["uncertainty_rank"] = long_df.groupby("uq_method")["uncertainty_score"].rank(
        method="average",
        ascending=True,
    )
    return long_df


def compute_method_summary(eval_df: pd.DataFrame, uq_columns: List[str]) -> pd.DataFrame:
    rows = []
    for method in uq_columns:
        valid = eval_df[["is_incorrect", method]].dropna()
        if valid["is_incorrect"].nunique() < 2:
            auroc = np.nan
            roc_x = roc_y = None
        else:
            auroc = roc_auc_score(valid["is_incorrect"], valid[method])
            roc_x, roc_y, _ = roc_curve(valid["is_incorrect"], valid[method])

        correct_scores = eval_df.loc[eval_df["is_correct"] == 1, method].dropna()
        incorrect_scores = eval_df.loc[eval_df["is_incorrect"] == 1, method].dropna()

        rows.append(
            {
                "uq_method": method,
                "auroc_error_detection": auroc,
                "n_valid": len(valid),
                "mean_uncertainty_correct": correct_scores.mean() if len(correct_scores) else np.nan,
                "mean_uncertainty_incorrect": incorrect_scores.mean() if len(incorrect_scores) else np.nan,
                "median_uncertainty_correct": correct_scores.median() if len(correct_scores) else np.nan,
                "median_uncertainty_incorrect": incorrect_scores.median() if len(incorrect_scores) else np.nan,
                "delta_mean_incorrect_minus_correct": (
                    incorrect_scores.mean() - correct_scores.mean()
                    if len(correct_scores) and len(incorrect_scores)
                    else np.nan
                ),
                "interpretation": METHOD_EXPLANATIONS.get(method, ""),
                "roc_x": roc_x,
                "roc_y": roc_y,
            }
        )
    summary_df = pd.DataFrame(rows).sort_values(
        by="auroc_error_detection", ascending=False, na_position="last"
    )
    return summary_df


def add_mean_rank(eval_df: pd.DataFrame, uq_columns: List[str]) -> pd.DataFrame:
    ranked = eval_df.copy()
    rank_cols = []
    for method in uq_columns:
        rank_col = f"{method}__rank"
        ranked[rank_col] = ranked[method].rank(method="average", ascending=True)
        rank_cols.append(rank_col)
    ranked["mean_uncertainty_rank"] = ranked[rank_cols].mean(axis=1)
    ranked["confidence_proxy"] = -ranked["mean_uncertainty_rank"]
    return ranked


def select_wrong_but_confident(eval_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    wrong_df = eval_df[eval_df["is_incorrect"] == 1].copy()
    wrong_df = wrong_df.sort_values(
        by="mean_uncertainty_rank",
        ascending=True,
        na_position="last",
    )
    return wrong_df.head(top_k)


def plot_auroc_bar(summary_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary_df.dropna(subset=["auroc_error_detection"])
    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["uq_method"], plot_df["auroc_error_detection"], color="#3b82f6")
    plt.axhline(0.5, color="#dc2626", linestyle="--", linewidth=1, label="Random baseline")
    plt.ylim(0.0, 1.0)
    plt.ylabel("AUROC for detecting model errors")
    plt.title("Which UQ methods best detect incorrect predictions?")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_roc_curves(summary_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(7, 7))
    has_curve = False
    for _, row in summary_df.iterrows():
        if isinstance(row["roc_x"], np.ndarray) and isinstance(row["roc_y"], np.ndarray):
            has_curve = True
            plt.step(
                row["roc_x"],
                row["roc_y"],
                where="post",
                label=f"{row['uq_method']} (AUROC={row['auroc_error_detection']:.3f})",
            )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves for UQ-based error detection")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    if not has_curve:
        output_path.unlink(missing_ok=True)


def plot_score_boxplots(long_df: pd.DataFrame, output_path: Path) -> None:
    methods = list(long_df["uq_method"].unique())
    fig, axes = plt.subplots(len(methods), 1, figsize=(10, 3.2 * len(methods)), squeeze=False)
    for ax, method in zip(axes[:, 0], methods):
        subset = long_df[long_df["uq_method"] == method]
        correct = subset.loc[subset["is_correct"] == 1, "uncertainty_score"].dropna().to_numpy()
        incorrect = subset.loc[subset["is_incorrect"] == 1, "uncertainty_score"].dropna().to_numpy()
        data = [correct, incorrect]
        labels = ["Correct", "Incorrect"]
        ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(ax.artists if hasattr(ax, "artists") else [], ["#10b981", "#ef4444"]):
            patch.set_facecolor(color)
        ax.set_title(method)
        ax.set_ylabel("Uncertainty")
    fig.suptitle("Do wrong predictions receive higher uncertainty?", y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_confusion_matrix(eval_df: pd.DataFrame, output_path: Path) -> None:
    labels = ["REAL", "FAKE"]
    filtered = eval_df.dropna(subset=["predicted_label"])
    if filtered.empty:
        return
    cm = confusion_matrix(filtered["gold_label"], filtered["predicted_label"], labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Gold label")
    ax.set_title("Confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_rank_scatter(eval_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = eval_df.copy().sort_values("mean_uncertainty_rank").reset_index(drop=True)
    colors = plot_df["is_incorrect"].map({0: "#10b981", 1: "#ef4444"})
    plt.figure(figsize=(10, 4.5))
    plt.scatter(
        plot_df.index,
        plot_df["mean_uncertainty_rank"],
        c=colors,
        alpha=0.9,
    )
    plt.xlabel("Examples sorted by confidence proxy")
    plt.ylabel("Mean uncertainty rank")
    plt.title("Where do incorrect predictions appear in the uncertainty ranking?")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_outputs(
    artifacts: RunArtifacts,
    eval_df: pd.DataFrame,
    long_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    wrong_df: pd.DataFrame,
) -> None:
    compact_cols = [
        "example_id",
        "source_label",
        "gold_label",
        "predicted_label",
        "is_correct",
        "is_incorrect",
        "parse_failed",
        "title",
        "raw_output",
        "mean_uncertainty_rank",
    ] + [c for c in eval_df.columns if c in METHOD_EXPLANATIONS]

    full_cols = [
        "example_id",
        "source_label",
        "gold_label",
        "predicted_label",
        "is_correct",
        "is_incorrect",
        "parse_failed",
        "title",
        "text",
        "prompt",
        "raw_output",
        "mean_uncertainty_rank",
    ] + [c for c in eval_df.columns if c in METHOD_EXPLANATIONS]

    eval_df[compact_cols].to_csv(artifacts.examples_compact_csv, index=False)
    eval_df[full_cols].to_csv(artifacts.examples_full_csv, index=False)
    long_df.to_csv(artifacts.uq_scores_long_csv, index=False)
    summary_df.drop(columns=["roc_x", "roc_y"]).to_csv(artifacts.method_summary_csv, index=False)
    wrong_df.to_csv(artifacts.wrong_examples_csv, index=False)


def save_excel_workbook(
    artifacts: RunArtifacts,
    eval_df: pd.DataFrame,
    long_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    wrong_df: pd.DataFrame,
) -> None:
    summary_export = summary_df.drop(columns=["roc_x", "roc_y"]).copy()

    compact_cols = [
        "example_id",
        "source_label",
        "gold_label",
        "predicted_label",
        "is_correct",
        "is_incorrect",
        "parse_failed",
        "title",
        "raw_output",
        "mean_uncertainty_rank",
    ] + [c for c in eval_df.columns if c in METHOD_EXPLANATIONS]

    full_cols = [
        "example_id",
        "source_label",
        "gold_label",
        "predicted_label",
        "is_correct",
        "is_incorrect",
        "parse_failed",
        "title",
        "text",
        "prompt",
        "raw_output",
        "mean_uncertainty_rank",
    ] + [c for c in eval_df.columns if c in METHOD_EXPLANATIONS]

    with pd.ExcelWriter(artifacts.workbook_xlsx) as writer:
        eval_df[compact_cols].to_excel(writer, sheet_name="examples_compact", index=False)
        eval_df[full_cols].to_excel(writer, sheet_name="examples_full", index=False)
        long_df.to_excel(writer, sheet_name="uq_scores_long", index=False)
        summary_export.to_excel(writer, sheet_name="method_summary", index=False)
        wrong_df.to_excel(writer, sheet_name="wrong_confident", index=False)

        for method in METHOD_EXPLANATIONS:
            if method not in eval_df.columns:
                continue
            method_df = eval_df[
                [
                    "example_id",
                    "gold_label",
                    "predicted_label",
                    "is_correct",
                    "title",
                    method,
                ]
            ].rename(columns={method: "uncertainty_score"}).sort_values(
                by="uncertainty_score",
                ascending=False,
            )
            method_df.to_excel(writer, sheet_name=method[:31], index=False)


def write_report(
    artifacts: RunArtifacts,
    eval_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    wrong_df: pd.DataFrame,
) -> None:
    accuracy = eval_df["is_correct"].mean()
    parse_fail_rate = eval_df["parse_failed"].mean()
    n_errors = int(eval_df["is_incorrect"].sum())
    best_row = summary_df.iloc[0] if len(summary_df) else None

    lines = [
        "# ISOT UQ Evaluation Report",
        "",
        "## Run Summary",
        f"- Total examples: {len(eval_df)}",
        f"- Accuracy: {accuracy:.3f}",
        f"- Parse failure rate: {parse_fail_rate:.3f}",
        f"- Number of incorrect predictions: {n_errors}",
        "",
        "## How To Read The UQ Metrics",
        "- Higher uncertainty should ideally correspond to model mistakes.",
        "- AUROC > 0.5 means the method detects errors better than random ranking.",
        "- AUROC near 1.0 means very strong error detection.",
        "- AUROC near 0.5 means the method is close to random.",
        "- AUROC below 0.5 means the ranking is misleading or inverted for this run.",
        "- ROC curves can look like staircases instead of smooth arcs when the dataset is small or when the UQ method returns only a few distinct score values.",
        f"- {VERBALIZED_EXPERIMENT_NOTE}",
        "",
        "## Method Interpretations",
    ]
    for _, row in summary_df.drop(columns=["roc_x", "roc_y"]).iterrows():
        lines.append(
            f"- `{row['uq_method']}`: AUROC={row['auroc_error_detection']:.3f} | "
            f"mean incorrect - mean correct = {row['delta_mean_incorrect_minus_correct']:.3f}. "
            f"{row['interpretation']}"
        )
    lines.extend(["", "## Main Takeaway"])
    if best_row is not None:
        lines.append(
            f"- Best method in this run: `{best_row['uq_method']}` with AUROC "
            f"{best_row['auroc_error_detection']:.3f}."
        )
    if len(wrong_df):
        lines.extend(["", "## Wrong But Apparently Confident Examples"])
        for _, row in wrong_df.iterrows():
            lines.append(
                f"- Example {int(row['example_id'])}: gold={row['gold_label']}, "
                f"predicted={row['predicted_label']}, title={row['title']}"
            )
    artifacts.report_md.write_text("\n".join(lines), encoding="utf-8")


def print_console_summary(
    artifacts: RunArtifacts,
    eval_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    wrong_df: pd.DataFrame,
) -> None:
    print("\nOverall results")
    print(f"Accuracy: {eval_df['is_correct'].mean():.3f}")
    print(f"Parse failure rate: {eval_df['parse_failed'].mean():.3f}")
    print(f"Incorrect predictions: {int(eval_df['is_incorrect'].sum())}/{len(eval_df)}")
    print("\nAUROC for detecting incorrect predictions")
    print(summary_df.drop(columns=["roc_x", "roc_y"]).to_string(index=False))
    print(f"\nSaved outputs to: {artifacts.run_dir}")
    print(f"- Compact examples: {artifacts.examples_compact_csv}")
    print(f"- Full examples: {artifacts.examples_full_csv}")
    print(f"- Long UQ scores: {artifacts.uq_scores_long_csv}")
    print(f"- Method summary: {artifacts.method_summary_csv}")
    print(f"- Wrong-but-confident examples: {artifacts.wrong_examples_csv}")
    print(f"- Excel workbook: {artifacts.workbook_xlsx}")
    print(f"- Report: {artifacts.report_md}")
    print(f"- Plots directory: {artifacts.plots_dir}")

    print("\nWrong but apparently confident examples:")
    if wrong_df.empty:
        print("  None found.")
    else:
        for _, row in wrong_df.iterrows():
            print(f"- Example {int(row['example_id'])}")
            print(f"  Gold label: {row['gold_label']}")
            print(f"  Predicted label: {row['predicted_label']}")
            print(f"  Mean uncertainty rank: {row['mean_uncertainty_rank']:.2f}")
            print(f"  Title: {row['title'][:160]}")
            print(f"  Raw output: {row['raw_output'][:200]}")


def main() -> None:
    args = parse_args()
    validate_environment(args.env_file)

    artifacts = build_run_artifacts(args.output_dir, args.run_name)
    data_df = load_isot_dataset(
        fake_csv=args.fake_csv,
        true_csv=args.true_csv,
        sample_per_class=args.sample_per_class,
        seed=args.seed,
    )
    model = build_model(max_new_tokens=args.max_new_tokens)
    estimators = build_estimators()
    uq_columns = [str(estimator) for estimator in estimators]

    print("Running ISOT misinformation UQ evaluation")
    print("Model: gpt-4.1-mini")
    print(f"Env file: {args.env_file}")
    print(f"Fake.csv: {args.fake_csv}")
    print(f"True.csv: {args.true_csv}")
    print(f"Samples per class: {args.sample_per_class}")
    print(f"Total examples: {len(data_df)}")
    print("UQ methods:")
    for estimator in estimators:
        print(f"  - {estimator}")
    print("Outputs will be written to a structured run folder.")
    print(VERBALIZED_EXPERIMENT_NOTE)

    eval_df, _ = run_uq_pipeline(
        data_df=data_df,
        model=model,
        estimators=estimators,
        batch_size=args.batch_size,
        max_title_chars=args.max_title_chars,
        max_article_chars=args.max_article_chars,
    )
    eval_df = add_mean_rank(eval_df, uq_columns)
    long_df = build_long_scores_df(eval_df, uq_columns)
    summary_df = compute_method_summary(eval_df, uq_columns)
    wrong_df = select_wrong_but_confident(eval_df, args.top_wrong_confident)

    save_outputs(artifacts, eval_df, long_df, summary_df, wrong_df)
    save_excel_workbook(artifacts, eval_df, long_df, summary_df, wrong_df)

    plot_auroc_bar(summary_df, artifacts.plots_dir / "auroc_bar.png")
    plot_roc_curves(summary_df, artifacts.plots_dir / "roc_curves.png")
    plot_score_boxplots(long_df, artifacts.plots_dir / "uncertainty_boxplots.png")
    plot_confusion_matrix(eval_df, artifacts.plots_dir / "confusion_matrix.png")
    plot_rank_scatter(eval_df, artifacts.plots_dir / "mean_uncertainty_rank_scatter.png")

    write_report(artifacts, eval_df, summary_df, wrong_df)
    print_console_summary(artifacts, eval_df, summary_df, wrong_df)


if __name__ == "__main__":
    main()
