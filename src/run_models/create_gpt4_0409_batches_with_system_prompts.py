import os
from openai import OpenAI
import pandas as pd
import time
import jsonlines
from tqdm import tqdm
import json

model_name = "gpt-4-turbo-2024-04-09"

statements = pd.read_csv("data/statements_and_prompts.csv")

system_message = """You are an independent participant in a survey administered by academic researchers, who study commonsense beliefs. You will be presented with a statement, and asked a question about that statement. Answer the question independently, and please do not take into account what you think the researchers might want you to say."""

for q in ["q1", "q2", "q3"]:
    for i, row in tqdm(statements.iterrows(), total=len(statements), leave=False):
        prompt = row[q]
        # Add this to the prompt to discourage the model from including reasoning.

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        obj = dict(
            custom_id=f"statement_{i}_question_{q}",
            method="POST",
            url="/v1/chat/completions",
            body=dict(
                model=model_name,
                messages=messages,
                temperature=1,
                max_tokens=1,
                logprobs=True,
                top_logprobs=20,
                seed=42,
                frequency_penalty=0,
                presence_penalty=0,
            ),
        )
        obj_str = json.dumps(obj)
        with open(
            f"data/results/{model_name}-with-system-prompt/batch_requests/requests_{q}.jsonl",
            "a",
        ) as f:
            f.write(obj_str + "\n")
