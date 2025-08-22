import json
import logging
import os
import random
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
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
RESULTS_ROOT = Path("results")


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
        logging.error("Error: Could not initialize OpenAI client. "
                      "Ensure OPENROUTER_API_KEY is set.")
        logging.error(f"Details: {e}")
        raise e


def sanitize_for_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def make_run_dir(results_root: Path = RESULTS_ROOT) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = results_root / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_results_path(model: str, run_dir: Path) -> Path:
    fname = f"{sanitize_for_filename(model)}_aime25.jsonl"
    return run_dir / fname


ANSWER_LINE_RE = re.compile(r"(?im)^\s*Answer:\s*(-?\d+)\s*$")
BOXED_RE = re.compile(r"\\boxed\\{(-?\\d+)\\}")


def extract_final_answer(text: str):
    if not text:
        return None
    m = ANSWER_LINE_RE.search(text) or BOXED_RE.search(text)
    if not m:
        return None
    try:
        val = int(m.group(1))
        return val if 0 <= val <= 999 else None
    except Exception:
        return None


def chat_with_retry(client: OpenAI, model: str, messages: list, max_retries: int = 4):
    last_exc = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(model=model, messages=messages)
        except Exception as e:
            last_exc = e
            # jittered exponential backoff
            delay = (1.5 ** attempt) + random.uniform(0, 0.5)
            logging.warning(f"[{model}] error attempt {attempt + 1}/{max_retries}: {e}. sleeping {delay:.2f}s")
            time.sleep(delay)
    raise RuntimeError(f"Permanent failure for model {model}") from last_exc


def evaluate_model_on_aime(model: str, run_dir: Path):
    logging.info(f"--- Starting evaluation for model: {model} ---")
    client = get_client()
    dataset = load_dataset(DATASET_FILE_PATH)
    if not dataset:
        logging.error(f"[{model}] Dataset empty; aborting.")
        return

    results_path = make_results_path(model, run_dir)
    logging.info(f"[{model}] Writing results to: {results_path.resolve()}")

    for i, item in enumerate(dataset, start=1):
        problem = item["problem"]
        expected_solution = item["solution"]
        logging.info(f"[{model}] Problem #{i}/{len(dataset)}")

        messages = [
            {"role": "system", "content": "You are a careful competition mathematician. Be concise and correct."},
            {"role": "user", "content":
                "Solve the problem step by step. Then, on a new final line, output exactly:\n"
                "Answer: <integer>\n\n"
                f"{problem}"
             },
        ]

        try:
            response = chat_with_retry(client, model, messages)
            row = {
                "idx": i,
                "model": model,
                "problem": problem,
                "expected_answer": int(expected_solution),
                "response": response.model_dump(),
            }
        except Exception as e:
            tb_str = traceback.format_exc()
            logging.error(f"[{model}] error on problem #{i}: {e}\n{tb_str}")
            row = {
                "idx": i,
                "model": model,
                "problem": problem,
                "expected_answer": int(expected_solution),
                "response": {"error": tb_str},
            }

        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_run_manifest(run_dir: Path, dataset_path: str, models: list[str], model_threads: int) -> None:
    manifest = {
        "run_timestamp": run_dir.name,
        "dataset": dataset_path,
        "models": models,
        "parallel_models": model_threads,
    }
    with open(run_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def evaluate_models_parallel(models: list[str], threads: int):
    """
    Run whole-model evaluations in parallel. Each model runs sequentially through the dataset
    but different models run concurrently (default 2).
    """
    run_dir = make_run_dir(RESULTS_ROOT)
    logging.info(f"Run directory: {run_dir.resolve()}")
    write_run_manifest(run_dir, DATASET_FILE_PATH, models, threads)

    with ThreadPoolExecutor(max_workers=max(1, threads)) as executor:
        futures = {executor.submit(evaluate_model_on_aime, m, run_dir): m for m in models}
        for fut in as_completed(futures):
            model = futures[fut]
            try:
                fut.result()
                logging.info(f"[{model}] completed")
            except Exception as e:
                logging.error(f"[{model}] crashed: {e}")


def evaluate_all_free_models(num_threads: int) -> None:
    evaluate_models_parallel(free_models, num_threads)


if __name__ == "__main__":
    # parallelize across models (default 2)
    evaluate_all_free_models(1)
