from tqdm import tqdm
import os
import pandas as pd
import jsonlines
import time
import json


from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os

api_key = os.environ["MISTRAL_AI_API"]
model = "mistral-large-latest"

statements = pd.read_csv("data/statements_and_prompts.csv")

client = MistralClient(api_key=api_key)

for trial_no in range(1, 24):

    trial_name = "" if trial_no == 1 else f"_trial{trial_no}"

    questions_unanswered = {q: set(range(len(statements))) for q in ["q1", "q2", "q3"]}
    for q in ["q1", "q2", "q3"]:
        if os.path.exists(
            f"data/results/mistral-large-latest/results_raw_{q}{trial_name}.jsonl"
        ):
            with open(
                f"data/results/mistral-large-latest/results_raw_{q}{trial_name}.jsonl",
                "r",
            ) as fp:
                for line in jsonlines.Reader(fp):
                    questions_unanswered[q].remove(line["id"])

    results = {"q1": [], "q2": [], "q3": []}

    for q in ["q1", "q2", "q3"]:
        for i, row in tqdm(statements.iterrows(), total=len(statements)):

            if i not in questions_unanswered[q]:
                continue

            prompt = row[q]

            messages = [ChatMessage(role="user", content=prompt)]

            try:
                response = client.chat(
                    model=model,
                    messages=messages,
                    temperature=1,
                    random_seed=42,
                    max_tokens=1,
                )
            except Exception as e:
                time.sleep(20)
                response = client.chat(
                    model=model,
                    messages=messages,
                    temperature=1,
                    random_seed=42,
                    max_tokens=1,
                )

            if i % 1000 == 0:
                time.sleep(10)

            with open(
                f"data/results/mistral-large-latest/results_raw_{q}{trial_name}.jsonl",
                "a",
            ) as f:
                f.write(
                    json.dumps(
                        {
                            "id": i,
                            "prompt": prompt,
                            "response": response.choices[0].message.content,
                        }
                    )
                    + "\n"
                )
