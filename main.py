import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

chimera_r1t2 = "tngtech/deepseek-r1t2-chimera:free"
chimera_r1t = "tngtech/deepseek-r1t-chimera:free"
deepseek_r1 = "deepseek/deepseek-r1:free"
deepseek_r1_0528 = "deepseek/deepseek-r1-0528:free"
deepseek_v3 = "deepseek/deepseek-chat-v3-0324:free"
# deepseek_v3_1 = "deepseek/deepseek-v3:free"
qwen3 = "qwen/qwen3-235b-a22b-thinking-2507"
moonshot_k2 = "moonshotai/kimi-k2:free"
google_gemma3 = "google/gemma-3-27b-it:free"
openai_gpt_oss_20b = "openai/gpt-oss-20b:free"
openai_gpt_oss_120b = "openai/gpt-oss-120b"
z_ai_glm_4_5 = "z-ai/glm-4.5"
mistral_nemo = "mistralai/mistral-nemo:free"
mistral_small_24b = "mistralai/mistral-small-3.2-24b-instruct:free"


free_models = [
    chimera_r1t2, chimera_r1t, deepseek_r1, deepseek_r1_0528, deepseek_v3,
    qwen3, moonshot_k2, google_gemma3, openai_gpt_oss_20b, z_ai_glm_4_5, mistral_nemo, mistral_small_24b,
]
payed_models = [openai_gpt_oss_120b]

DATASET_FILE_PATH = "benchmarks/aime-25.jsonl"
RESULTS_ROOT = Path("results")  # parent directory for all runs


def load_dataset(filepath):
    dataset = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                dataset.append({"problem": data["problem"], "solution": data["answer"]})
        return dataset
    except FileNotFoundError:
        logging.error(f"Error: Dataset file not found at '{filepath}'")
        return []
    except Exception as e:
        logging.error(f"Error loading or parsing dataset file: {e}")
        return []


def get_client() -> OpenAI:
    try:
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    except Exception as e:
        logging.error("Error: Could not initialize OpenAI client.")
        logging.error(
            "Please make sure your OPENROUTER_API_KEY environment variable is set correctly."
        )
        logging.error(f"Details: {e}")
        raise e


def sanitize_for_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def make_run_dir(results_root: Path = RESULTS_ROOT) -> Path:
    """
    Create and return a timestamped directory for this run, e.g.:
    results/20250820-153541/
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = results_root / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_results_path(model: str, run_dir: Path) -> Path:
    """
    Build the file path for a model's results inside the given run directory.
    Example: results/20250820-153541/tngtech_deepseek-r1t2-chimera_free_aime25.jsonl
    """
    fname = f"{sanitize_for_filename(model)}_aime25.jsonl"
    return run_dir / fname


ANSWER_LINE_RE = re.compile(r"(?im)^\s*Answer:\s*(-?\d+)\s*$")
BOXED_RE = re.compile(r"\\boxed\\{(-?\\d+)\\}")

def extract_final_answer(text: str):
    """
    Prefer 'Answer: <int>' on its own line; fall back to \boxed{<int>}.
    Enforce AIME integer range [0, 999].
    """
    if not text:
        return None

    m = ANSWER_LINE_RE.search(text)
    if not m:
        m = BOXED_RE.search(text)
    if not m:
        return None

    try:
        val = int(m.group(1))
        return val if 0 <= val <= 999 else None
    except Exception:
        return None


def evaluate_model_on_aime(model: str, run_dir: Path):
    logging.info(f"--- Starting evaluation for model: {model} ---")

    client = get_client()
    dataset = load_dataset(DATASET_FILE_PATH)
    if not dataset:
        logging.error("Dataset empty; aborting.")
        return

    results_path = make_results_path(model, run_dir)
    logging.info(f"Writing per-problem results to: {results_path.resolve()}")


    for i, item in enumerate(dataset):
        problem = item["problem"]
        expected_solution = item["solution"]
        logging.info(f"\n----- Evaluating Problem #{i + 1} of {len(dataset)} -----")
        logging.info(f"Problem: {problem}")

        try:
            prompt_messages = [
                {
                    "role": "system",
                    "content": "You are a careful competition mathematician. Be concise and correct.",
                },
                {
                    "role": "user",
                    "content": (
                        "Solve the problem step by step. Then, on a new final line, output exactly:\n"
                        "Answer: <integer>\n\n"
                        f"{problem}"
                    ),
                },
            ]

            response = client.chat.completions.create(
                model=model,
                messages=prompt_messages,
                # temperature=0.2,  # optional: keep outputs tighter
            )        


            # --- Write one JSONL row per problem ---
            row = {
                "idx": i + 1,
                "model": model,
                "problem": problem,
                "expected_answer": int(expected_solution),
                "response": response.model_dump(),
            }
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        except Exception as e:
            logging.error(f"An error occurred while processing problem #{i + 1}: {e} and model: {model}")
            


def write_run_manifest(run_dir: Path, dataset_path: str, models: list[str]) -> None:
    """
    Optional: saves basic run metadata alongside model files.
    """
    manifest = {
        "run_timestamp": run_dir.name,  # matches directory name
        "dataset": dataset_path,
        "models": models,
    }
    with open(run_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def evaluate_all_free_models():
    # Create a fresh subdirectory for this run
    run_dir = make_run_dir(RESULTS_ROOT)
    logging.info(f"Run directory: {run_dir.resolve()}")

    write_run_manifest(run_dir, DATASET_FILE_PATH, free_models)

    for model in free_models:
        evaluate_model_on_aime(model, run_dir)


if __name__ == "__main__":
    evaluate_all_free_models()
