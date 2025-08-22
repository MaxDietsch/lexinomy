from pathlib import Path
import json
from main import chat_with_retry, get_client, load_dataset
import logging 
DATASET_FILE_PATH = "benchmarks/aime-25.jsonl"

def retry_failed_problems(file):
    with Path(file).open("r", encoding="utf-8") as f: 
        data = [json.loads(line) for line in f]

    dataset = load_dataset(DATASET_FILE_PATH)

    rows = []
    for id, entry in enumerate(data):
        row = entry 
        if "error" in entry["response"].keys():
            problem = dataset[id]["problem"]
            client = get_client() 
            expected_solution = dataset[id]["solution"]
            model = entry["model"]

            logging.info(f"[{model}] Problem #{id}/{len(dataset)} will be retried")

            messages = [
                {"role": "system", "content": "You are a careful competition mathematician. Be concise and correct."},
                {"role": "user", "content":
                    "Solve the problem step by step. Then, on a new final line, output exactly:\n"
                    "Answer: \n\n"
                    f"{problem}"
                },
            ]

            try:
                response = chat_with_retry(client, model, messages)
                row = {
                    "idx": id,
                    "model": model,
                    "problem": problem,
                    "expected_answer": int(expected_solution),
                    "response": response.model_dump(),
                }
            except Exception as e:
                logging.error(f"[{model}] error on problem #{i}: {e}")
                row = {
                    "idx": id,
                    "model": model,
                    "problem": problem,
                    "expected_answer": int(expected_solution),
                    "response": {"error": str(e)},
                }

        rows.append(row)
    with open(f"{file}.fixed", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    dir = Path("results/20250822-075251")

    files = sorted(dir.glob("*.jsonl"))
    for file in files:
        retry_failed_problems(file)