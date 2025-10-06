from tqdm import tqdm
import os
from openai import OpenAI
import numpy as np
import pandas as pd
import pickle
import jsonlines
import time
import json


# Connect to OpenAI
OPENAI_API = os.getenv("OPENAI_API")
if OPENAI_API is None:
    OPENAI_API = input("Input your OpenAI API key:")
client = OpenAI(api_key=OPENAI_API)

statements = pd.read_csv("data/statements_and_prompts.csv")

results = {"q1": [], "q2": [], "q3": []}
# Models: "gpt-4-0124-preview", "gpt-4o-2024-05-13", "gpt-4-turbo-2024-0409"
model_name = "gpt-4o-2024-05-13"

for trial_no in range(1, 6):
    trial_name = "" if trial_no == 1 else f"_trial{trial_no}"
    questions_unanswered = {q: set(range(len(statements))) for q in ["q1", "q2", "q3"]}
    for q in ["q1", "q2", "q3"]:
        if os.path.exists(
            f"data/results/gpt-4o-2024-05-13/results_raw_{q}{trial_name}.jsonl"
        ):
            with open(
                f"data/results/gpt-4o-2024-05-13/results_raw_{q}{trial_name}.jsonl", "r"
            ) as fp:
                for line in jsonlines.Reader(fp):
                    questions_unanswered[q].remove(line["id"])

    for q in ["q1", "q2", "q3"]:
        for i, row in tqdm(statements.iterrows(), total=len(statements)):

            if i not in questions_unanswered[q]:
                continue

            messages = [{"role": "user", "content": row[q]}]

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=1,
                max_tokens=1,
                logprobs=True,
                top_logprobs=20,
                seed=42,
                frequency_penalty=0,
                presence_penalty=0,
            )

            results[q].append({"id": i, "prompt": row[q], "response": response.dict()})

            if i % 1000 == 0:
                time.sleep(10)

            with open(
                f"data/results/gpt-4o-2024-05-13/results_raw_{q}{trial_name}.jsonl", "a"
            ) as f:
                f.write(
                    json.dumps({"id": i, "prompt": row[q], "response": response.dict()})
                    + "\n"
                )
