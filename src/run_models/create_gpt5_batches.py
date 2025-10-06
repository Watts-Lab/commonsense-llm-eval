import os
from openai import OpenAI
import pandas as pd
import time
import jsonlines
from tqdm import tqdm
import json

model_name = "gpt-5-2025-08-07"

statements = pd.read_csv("data/statements_and_prompts.csv")

for q in ["q1", "q2", "q3"]:
    for i, row in tqdm(statements.iterrows(), total=len(statements), leave=False):
        prompt = row[q]
        # Add this to the prompt to discourage the model from including reasoning.
        prompt += " Do not include anything else, such as an explanation or reasoning."

        messages = [{"role": "user", "content": prompt}]

        for request_id in range(1, 7):
            obj = dict(
                custom_id=f"statement_{i}_question_{q}_request_{request_id}",
                method="POST",
                url="/v1/chat/completions",
                body=dict(
                    model=model_name,
                    messages=messages,
                    temperature=1,
                    max_completion_tokens=100,
                    # seed=42,
                    reasoning_effort="minimal",
                    frequency_penalty=0,
                    presence_penalty=0,
                    n=8,
                ),
            )
            obj_str = json.dumps(obj)
            with open(
                f"data/results/{model_name}/batch_requests/requests_{q}.jsonl", "a"
            ) as f:
                f.write(obj_str + "\n")
