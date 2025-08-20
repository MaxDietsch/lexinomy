import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

MODEL_TO_EVALUATE = "tngtech/deepseek-r1t2-chimera:free"
DATASET_FILE_PATH = "benchmarks/aime-25.jsonl"


def load_dataset(filepath):
    dataset = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                dataset.append({
                    "problem": data["problem"],
                    "solution": data["answer"]
                })
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
            api_key=os.getenv('OPENROUTER_API_KEY')
        )
    except Exception as e:
        logging.error("Error: Could not initialize OpenAI client.")
        logging.error("Please make sure your OPENROUTER_API_KEY environment variable is set correctly.")
        logging.error(f"Details: {e}")
        raise e


def evaluate_model_on_aime(model):
    logging.info(f"--- Starting evaluation for model: {model} ---")

    client = get_client()
    dataset = load_dataset(DATASET_FILE_PATH)

    total_correct = 0
    total_output_tokens = 0

    for i, item in enumerate(dataset):
        problem = item["problem"]
        expected_solution = str(item["solution"])
        logging.info(f"\n----- Evaluating Problem #{i + 1} -----")
        logging.debug(f"Problem: {problem}")

        try:
            # TODO: play around with system prompt, it is necessary at all? Maybe prompt the model to just return the solution.
            prompt_messages = [
                {
                    "role": "system",
                    "content": "You are a world-class mathematician. Your task is to solve the following problem. Provide your reasoning, but always end your response with 'The final answer is ###', where ### is the integer solution.",
                },
                {
                    "role": "user",
                    "content": problem,
                },
            ]

            response = client.chat.completions.create(
                model=model,
                messages=prompt_messages,
            )

            logging.debug(f"Answer: {response}")
            model_answer = response.choices[0].message.content
            output_tokens = response.usage.completion_tokens

            logging.debug(f"Model's Raw Output:\n{model_answer}")
            logging.debug(f"Output Tokens: {output_tokens}")

            is_correct = f" {expected_solution} " in f" {model_answer} " or model_answer.endswith(
                expected_solution)  # TODO eval logic

            if is_correct:
                total_correct += 1
                logging.info("Answer was: CORRECT")
            else:
                logging.info(
                    f"Answer was: INCORRECT (Expected to find: {expected_solution}, but solution is {model_answer})")

            total_output_tokens += output_tokens

        except Exception as e:
            logging.error(f"An error occurred while processing problem #{i + 1}: {e}")

    # --- Final Results ---
    logging.info("\n\n===== Evaluation Summary =====")
    num_problems = len(dataset)
    accuracy = (total_correct / num_problems) * 100 if num_problems > 0 else 0
    avg_tokens = total_output_tokens / num_problems if num_problems > 0 else 0

    logging.info(f"Model: {model}")
    logging.info(f"Total Problems Evaluated: {num_problems}")
    logging.info(f"Correct Answers: {total_correct}")
    logging.info(f"Accuracy: {accuracy:.2f}%")
    logging.info("---------------------------------")
    logging.info(f"Total Output Tokens Used: {total_output_tokens}")
    logging.info(f"Average Output Tokens per Problem: {avg_tokens:.2f}")


if __name__ == "__main__":
    model = MODEL_TO_EVALUATE
    evaluate_model_on_aime(model)
