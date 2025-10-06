from tqdm import tqdm
import os
import anthropic
import numpy as np
import pandas as pd
import pickle
import jsonlines
import time
import json


# Connect to Anthropic
ANTHROPIC_API = os.getenv("ANTHROPIC_API")
if ANTHROPIC_API is None:
    ANTHROPIC_API = input("Input your Anthropic API key:")

# Models: "claude-3-haiku-20240307", "claude-3-sonnet-20240307", "claude-3-opus-20240307"
model_name = "claude-3-haiku-20240307"

client = anthropic.Client(api_key=ANTHROPIC_API)

statements = pd.read_csv("data/statements_and_prompts.csv")

for trial_no in range(1, 24):

    trial_name = "" if trial_no == 1 else f"_trial{trial_no}"

    questions_unanswered = {q: set(range(len(statements))) for q in ["q1", "q2", "q3"]}
    for q in ["q1", "q2", "q3"]:
        if os.path.exists(
            f"data/results/claude-3-haiku/results_raw_{q}{trial_name}.jsonl"
        ):
            with open(
                f"data/results/claude-3-haiku/results_raw_{q}{trial_name}.jsonl", "r"
            ) as fp:
                for line in jsonlines.Reader(fp):
                    questions_unanswered[q].remove(line["id"])

    results = {"q1": [], "q2": [], "q3": []}
    model_name = "claude-3-haiku-20240307"

    for q in ["q1", "q2", "q3"]:
        for i, row in tqdm(statements.iterrows(), total=len(statements)):

            if i not in questions_unanswered[q]:
                continue

            prompt = row[q]

            messages = [{"role": "user", "content": prompt}]

            try:
                response = client.messages.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1,
                    temperature=1,
                )
            except Exception as e:
                time.sleep(20)
                response = client.messages.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1,
                    temperature=1,
                )

            if i % 1000 == 0:
                time.sleep(10)

            with open(
                f"data/results/claude-3-haiku/results_raw_{q}{trial_name}.jsonl", "a"
            ) as f:
                f.write(
                    json.dumps(
                        {
                            "id": i,
                            "prompt": prompt,
                            "response": response.content[0].text,
                        }
                    )
                    + "\n"
                )
