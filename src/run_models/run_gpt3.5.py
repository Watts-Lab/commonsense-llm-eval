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

questions_unanswered = {q: set(range(len(statements))) for q in ["q1", "q2", "q3"]}
for q in ["q1", "q2", "q3"]:
    if os.path.exists(f"data/results/gpt3.5/results_raw_{q}_trial5.jsonl"):
        with open(f"data/results/gpt3.5/results_raw_{q}_trial5.jsonl", "r") as fp:
            for line in jsonlines.Reader(fp):
                questions_unanswered[q].remove(line["id"])

results = {"q1": [], "q2": [], "q3": []}
model_name = "gpt-3.5-turbo-0125"

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
            top_logprobs=5,
            seed=42,
            frequency_penalty=0,
            presence_penalty=0,
        )

        results[q].append({"id": i, "prompt": row[q], "response": response.dict()})

        if i % 1000 == 0:
            time.sleep(10)

        with open(f"data/results/gpt3.5/results_raw_{q}_trial5.jsonl", "a") as f:
            f.write(
                json.dumps({"id": i, "prompt": row[q], "response": response.dict()})
                + "\n"
            )

    # with jsonlines.open(f"data/results/gpt3.5/results_raw_{q}.jsonl", "w") as writer:
    #     writer.write_all(results[q])
