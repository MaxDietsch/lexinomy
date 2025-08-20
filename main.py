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


free_models = [chimera_r1t2, deepseek_r1, deepseek_r1_0528, deepseek_v3, qwen3, moonshot_k2, google_gemma3, openai_gpt_oss_20b, z_ai_glm_4_5]
payed_models = [openai_gpt_oss_120b]

DATASET_FILE_PATH = "benchmarks/aime-25.jsonl"
RESULTS_DIR = "results"


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


def make_results_path(model: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{ts}_{sanitize_for_filename(model)}_aime25.jsonl"
    outdir = Path(RESULTS_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / fname


_ANSWER_RE = re.compile(r"The final answer is\s*([\-]?\d+)\b", re.IGNORECASE)


def extract_final_answer(text: str):
    """
    Extracts the integer after 'The final answer is ...'.
    Returns an int if found, else None.
    """
    if not text:
        return None
    m = _ANSWER_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def evaluate_model_on_aime(model: str):
    logging.info(f"--- Starting evaluation for model: {model} ---")

    client = get_client()
    dataset = load_dataset(DATASET_FILE_PATH)
    if not dataset:
        logging.error("Dataset empty; aborting.")
        return

    results_path = make_results_path(model)
    logging.info(f"Writing per-problem results to: {results_path.resolve()}")

    total_correct = 0
    total_output_tokens = 0
    num_problems = len(dataset)

    for i, item in enumerate(dataset):
        problem = item["problem"]
        expected_solution = item["solution"]
        logging.info(f"\n----- Evaluating Problem #{i + 1} of {num_problems} -----")
        logging.info(f"Problem: {problem}")

        try:
            prompt_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a world-class mathematician. Your task is to solve the following problem. "
                        "Provide your reasoning, but always end your response with 'The final answer is ###', "
                        "where ### is the integer solution."
                    ),
                },
                {"role": "user", "content": problem},
            ]

            response = client.chat.completions.create(
                model=model,
                messages=prompt_messages,
                # temperature=0.2,  # optional: keep outputs tighter
            )

            model_output = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None) or {}
            prompt_tokens = getattr(usage, "prompt_tokens", None) or usage.get("prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", None) or usage.get("completion_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", None) or usage.get("total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens)

            logging.info(f"Model's Raw Output:\n{model_output}")
            logging.info(f"Tokens — prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}")

            extracted_answer = extract_final_answer(model_output)
            is_correct = (extracted_answer is not None) and (int(extracted_answer) == int(expected_solution))

            if is_correct:
                total_correct += 1
                logging.info("Answer was: CORRECT")
            else:
                logging.info(
                    f"Answer was: INCORRECT (Expected: {expected_solution}, Extracted: {extracted_answer})"
                )

            total_output_tokens += completion_tokens

            # --- Write one JSONL row per problem ---
            row = {
                "idx": i + 1,
                "model": model,
                "problem": problem,
                "expected_answer": int(expected_solution),
                "model_output_raw": model_output,
                "extracted_final_answer": extracted_answer,
                "is_correct": is_correct,
                "tokens": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        except Exception as e:
            logging.error(f"An error occurred while processing problem #{i + 1}: {e}")
            # still write a row so you don't lose traceability
            row = {
                "idx": i + 1,
                "model": model,
                "problem": problem,
                "expected_answer": int(expected_solution),
                "error": str(e),
            }
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # --- Final Results ---
    logging.info("\n\n===== Evaluation Summary =====")
    accuracy = (total_correct / num_problems) * 100 if num_problems > 0 else 0
    avg_completion_tokens = total_output_tokens / num_problems if num_problems > 0 else 0

    logging.info(f"Model: {model}")
    logging.info(f"Total Problems Evaluated: {num_problems}")
    logging.info(f"Correct Answers: {total_correct}")
    logging.info(f"Accuracy: {accuracy:.2f}%")
    logging.info("---------------------------------")
    logging.info(f"Total Completion Tokens Used: {total_output_tokens}")
    logging.info(f"Average Completion Tokens per Problem: {avg_completion_tokens:.2f}")
    logging.info(f"Per-problem results saved to: {results_path.resolve()}")



def evaluate_all_free_models():
    for model in free_models:
        evaluate_model_on_aime(model)


if __name__ == "__main__":
    evaluate_all_free_models()
