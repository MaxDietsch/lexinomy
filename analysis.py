# aime_aggregate.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt


# ---------- Data loading ----------

def load_results(paths: Sequence[str | Path]) -> pd.DataFrame:
    """
    Load one or more JSONL result files created by your evaluator.
    Returns a flat DataFrame with columns:
      ['idx','model','problem','expected_answer','model_output_raw',
       'extracted_final_answer','is_correct','prompt_tokens',
       'completion_tokens','total_tokens','error','source_file']
    """
    rows: List[dict] = []
    for p in map(Path, paths):
        if not p.exists():
            raise FileNotFoundError(f"Results file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                tokens = rec.get("tokens", {}) or {}
                rows.append({
                    "idx": rec.get("idx"),
                    "model": rec.get("model"),
                    "problem": rec.get("problem"),
                    "expected_answer": rec.get("expected_answer"),
                    "model_output_raw": rec.get("model_output_raw"),
                    "extracted_final_answer": rec.get("extracted_final_answer"),
                    "is_correct": rec.get("is_correct"),
                    "prompt_tokens": tokens.get("prompt_tokens", 0) or 0,
                    "completion_tokens": tokens.get("completion_tokens", 0) or 0,
                    "total_tokens": tokens.get("total_tokens", 0) or 0,
                    "error": rec.get("error"),
                    "source_file": str(p),
                })
    df = pd.DataFrame(rows)

    # Defensive typing
    if "expected_answer" in df:
        df["expected_answer"] = pd.to_numeric(df["expected_answer"], errors="coerce")
    if "extracted_final_answer" in df:
        df["extracted_final_answer"] = pd.to_numeric(df["extracted_final_answer"], errors="coerce")
    for col in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Prefer recorded is_correct; otherwise recompute, then treat missing as False
    recomputed = df["extracted_final_answer"] == df["expected_answer"]
    df["is_correct_final"] = df["is_correct"].where(df["is_correct"].notna(), recomputed)
    df["is_correct_final"] = df["is_correct_final"].fillna(False).astype(bool)
    return df


def load_results_from_dir(results_dir: str | Path) -> pd.DataFrame:
    """
    Load all *.jsonl files from a results directory.
    """
    dir_path = Path(results_dir)
    files = sorted(dir_path.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found in: {dir_path}")
    return load_results(files)


# ---------- Metrics ----------

@dataclass(frozen=True)
class BasicStats:
    model: str
    n: int
    correct: int
    accuracy: float
    avg_prompt_tokens: float
    avg_completion_tokens: float
    avg_total_tokens: float
    sum_prompt_tokens: int
    sum_completion_tokens: int
    sum_total_tokens: int
    extraction_rate: float  # % of rows with a parsed final answer
    error_rate: float  # % of rows with an 'error' recorded


def compute_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-model stats. Returns a DataFrame with one row per model.
    """
    # Helper columns
    df = df.copy()
    df["has_extracted"] = df["extracted_final_answer"].notna()
    df["has_error"] = df["error"].notna()

    grouped = df.groupby("model", dropna=False)

    stats_rows: List[BasicStats] = []
    for model, g in grouped:
        n = len(g)
        correct = int(g["is_correct_final"].sum())
        acc = (correct / n * 100.0) if n else 0.0

        avg_pt = g["prompt_tokens"].mean() if n else 0.0
        avg_ct = g["completion_tokens"].mean() if n else 0.0
        avg_tt = g["total_tokens"].mean() if n else 0.0

        sum_pt = int(g["prompt_tokens"].sum())
        sum_ct = int(g["completion_tokens"].sum())
        sum_tt = int(g["total_tokens"].sum())

        extraction_rate = (g["has_extracted"].mean() * 100.0) if n else 0.0
        error_rate = (g["has_error"].mean() * 100.0) if n else 0.0

        stats_rows.append(BasicStats(
            model=model,
            n=n,
            correct=correct,
            accuracy=acc,
            avg_prompt_tokens=avg_pt,
            avg_completion_tokens=avg_ct,
            avg_total_tokens=avg_tt,
            sum_prompt_tokens=sum_pt,
            sum_completion_tokens=sum_ct,
            sum_total_tokens=sum_tt,
            extraction_rate=extraction_rate,
            error_rate=error_rate,
        ))

    return pd.DataFrame([s.__dict__ for s in stats_rows]).sort_values("accuracy", ascending=False)


def per_problem_answers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot table: rows = problem idx, columns = model, values = extracted_final_answer.
    Useful to inspect disagreements.
    """
    return df.pivot_table(
        index="idx", columns="model", values="extracted_final_answer", aggfunc="first"
    ).sort_index()


def per_problem_correctness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot table: rows = problem idx, columns = model, values = is_correct_final (bool).
    """
    return df.pivot_table(
        index="idx", columns="model", values="is_correct_final", aggfunc="first"
    ).sort_index()


# ---------- Visualization (Matplotlib only) ----------

def plot_accuracy_bar(stats_df: pd.DataFrame, rotate: int = 20) -> None:
    """
    Bar chart of accuracy (%) by model.
    """
    x = stats_df["model"]
    y = stats_df["accuracy"]
    plt.figure()
    plt.bar(x, y)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Model")
    plt.title("AIME-25 Accuracy by Model")
    plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout()
    plt.show()


def plot_avg_tokens_bar(stats_df: pd.DataFrame, rotate: int = 20) -> None:
    """
    Bar chart of average total tokens per problem by model.
    """
    x = stats_df["model"]
    y = stats_df["avg_total_tokens"]
    plt.figure()
    plt.bar(x, y)
    plt.ylabel("Avg Total Tokens / Problem")
    plt.xlabel("Model")
    plt.title("Average Total Tokens by Model")
    plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout()
    plt.show()


def plot_token_breakdown(stats_df: pd.DataFrame, rotate: int = 20) -> None:
    """
    Stacked bars (prompt vs completion) of average tokens per problem by model.
    """
    x = stats_df["model"]
    pt = stats_df["avg_prompt_tokens"]
    ct = stats_df["avg_completion_tokens"]

    plt.figure()
    plt.bar(x, pt, label="Prompt")
    plt.bar(x, ct, bottom=pt, label="Completion")
    plt.ylabel("Avg Tokens / Problem")
    plt.xlabel("Model")
    plt.title("Average Token Breakdown by Model")
    plt.xticks(rotation=rotate, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_completion_token_box(df: pd.DataFrame, rotate: int = 20) -> None:
    """
    Boxplot of completion tokens per model (distribution across problems).
    """
    data = [g["completion_tokens"].values for _, g in df.groupby("model")]
    labels = [m for m, _ in df.groupby("model")]
    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("Completion Tokens")
    plt.xlabel("Model")
    plt.title("Completion Token Distribution by Model")
    plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout()
    plt.show()


# ---------- Convenience entry point ----------

def summarize_from_dir(results_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all JSONL results from a directory and return:
      (full_df, stats_df)
    """
    full_df = load_results_from_dir(results_dir)
    stats_df = compute_basic_stats(full_df)
    return full_df, stats_df


def _shorten_model_label(model: str) -> str:
    """
    Compact label for plotting: strip provider prefix and ':free' suffix.
    """
    short = model.split("/")[-1] if "/" in model else model
    return short.replace(":free", "")


def plot_tokens_vs_accuracy_for_run(
        run_sub_dir: str | Path,
        token_metric: str = "sum_completion_tokens",  # or "sum_total_tokens"
        annotate: bool = True,
        save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Scatter plot: X = total tokens (per model over the run), Y = accuracy (%).
    - run_sub_dir: the timestamped directory under 'results/', e.g. 'results/20250820-153541'
    - token_metric: 'sum_completion_tokens' (default) or 'sum_total_tokens'
    - annotate: write model labels near points
    - save_path: optional path to save the figure (PNG). If None, only shows the plot.

    Returns the per-model stats DataFrame used for plotting.
    """
    run_sub_dir = Path(run_sub_dir)
    if not run_sub_dir.exists() or not run_sub_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_sub_dir}")

    # Load only this run's JSONL files
    files = sorted(run_sub_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL results in: {run_sub_dir}")

    full_df = load_results(files)  # your existing loader that accepts a list/sequence
    stats_df = compute_basic_stats(full_df)

    if token_metric not in {"sum_completion_tokens", "sum_total_tokens"}:
        raise ValueError("token_metric must be 'sum_completion_tokens' or 'sum_total_tokens'")

    x = stats_df[token_metric]
    y = stats_df["accuracy"]

    plt.figure()
    plt.scatter(x, y)
    plt.xlabel("Total completion tokens (run)" if token_metric == "sum_completion_tokens"
               else "Total tokens (run)")
    plt.ylabel("Accuracy (%) on AIME-25")
    plt.title(f"Tokens vs. Accuracy — {run_sub_dir.name}")
    plt.grid(True, alpha=0.3)

    if annotate:
        for _, row in stats_df.iterrows():
            plt.annotate(
                _shorten_model_label(row["model"]),
                (row[token_metric], row["accuracy"]),
                textcoords="offset points",
                xytext=(5, 5),
            )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()

    return stats_df


if __name__ == "__main__":
    run = "20250820-190320"
    plot_tokens_vs_accuracy_for_run(f"results/{run}",
                                    token_metric="sum_completion_tokens",
                                    annotate=True,
                                    save_path=f"results/{run}/tokens_vs_accuracy.png")
