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
        with p.open("r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        
        for entry in data:
            content = entry['response']['choices'][0]['message']['content']
            rows.append({
                'id': entry['idx'],
                'model': entry['model'],
                'is_correct': str(entry['expected_answer']) in content,
                'output_tokens': entry['response']['usage']['completion_tokens']
            })
    df = pd.DataFrame(rows)

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
    output_tokens: int 



def compute_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-model stats. Returns a DataFrame with one row per model.
    """
    # Helper columns
    df = df.copy()

    grouped = df.groupby("model", dropna=False)

    stats_rows: List[BasicStats] = []
    for model, g in grouped:
        n = len(g)
        correct = int(g["is_correct"].sum())
        acc = (correct / n * 100.0) if n else 0.0
        sum_ot = int(g["output_tokens"].sum())


        stats_rows.append(BasicStats(
            model=model,
            n=n,
            correct=correct,
            accuracy=acc,
            output_tokens = sum_ot,
        ))


    return pd.DataFrame([s.__dict__ for s in stats_rows]).sort_values("accuracy", ascending=False)



def _shorten_model_label(model: str) -> str:
    """
    Compact label for plotting: strip provider prefix and ':free' suffix.
    """
    short = model.split("/")[-1] if "/" in model else model
    return short.replace(":free", "")


def plot_tokens_vs_accuracy_for_run(
        stats_df: pd.DataFrame,
        annotate: bool = True,
        save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Scatter plot: X = total tokens (per model over the run), Y = accuracy (%).
    - stats_df: DataFrame from compute_basic_stats
    - token_metric: 'sum_completion_tokens' (default) or 'sum_total_tokens'
    - annotate: write model labels near points
    - save_path: optional path to save the figure (PNG). If None, only shows the plot.

    Returns the per-model stats DataFrame used for plotting.
    """


    x = stats_df["output_tokens"]
    y = stats_df["accuracy"]

    plt.figure()
    plt.scatter(x, y)
    plt.xlabel("Total completion tokens")
    plt.ylabel("Accuracy (%) on AIME-25")
    plt.title(f"Tokens vs. Accuracy ")
    plt.grid(True, alpha=0.3)

    if annotate:
        for _, row in stats_df.iterrows():
            plt.annotate(
                _shorten_model_label(row["model"]),
                (row["output_tokens"], row["accuracy"]),
                textcoords="offset points",
                xytext=(5, 5),
            )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()

    return stats_df


if __name__ == "__main__":
    dir = Path("results/20250821-233413")

    files = sorted(dir.glob("*.jsonl"))
    full_df = load_results(files)  # your existing loader that accepts a list/sequence
    print(full_df.head())

    stats_df = compute_basic_stats(full_df)
    print(stats_df)


    plot_tokens_vs_accuracy_for_run(stats_df,
                                    annotate=True,
                                    save_path=f"{dir}/tokens_vs_accuracy.png")
