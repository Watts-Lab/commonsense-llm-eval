from transformers import AutoTokenizer, T5ForConditionalGeneration
import numpy as np
import json
import pandas as pd
import os
import jsonlines
from tqdm import tqdm
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# Load statements
statements = pd.read_csv("data/statements_and_prompts.csv")

# Models: "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-xl", "google/flan-t5-xxl"
MODEL_NAME = "google/flan-xxl"
MODEL_DIRNAME = MODEL_NAME.replace("/", "--")
checkpoint_path = ""

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# with init_empty_weights():
#     model = model = T5ForConditionalGeneration.from_pretrained(
#         MODEL_NAME,
#         trust_remote_code=True
#     )

model = T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
)

model.eval()
model.tie_weights()

# Load token to answer mapping
with open(f"data/results/{MODEL_DIRNAME}/answer2tokid.json", "r") as f:
    answer2tokid = json.load(f)

questions_unanswered = {q: set(range(len(statements))) for q in ["q1", "q2", "q3"]}
for q in ["q1", "q2", "q3"]:
    if os.path.exists(f"data/results/{MODEL_DIRNAME}/results_raw_{q}.jsonl"):
        with open(f"data/results/{MODEL_DIRNAME}/results_raw_{q}.jsonl", "r") as fp:
            for line in jsonlines.Reader(fp):
                questions_unanswered[q].remove(line["id"])

for q in ["q1", "q2", "q3"]:
    for i, row in tqdm(statements.iterrows(), total=len(statements)):
        if i not in questions_unanswered[q]:
            continue

        # Encode the message
        inputs = tokenizer(row[q], return_tensors="pt").to(model.device)

        # Feedforward the input
        outputs = model.generate(
            **inputs,
            output_scores=True,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=False,
        )

        # The the probability of next token
        probs = torch.softmax(outputs.scores[0][0], dim=-1).detach().cpu().numpy()

        # Map token ids to answers
        answer_probs = {}
        answer_probs["yes"] = float(probs[answer2tokid["yes"]].sum())
        answer_probs["no"] = float(probs[answer2tokid["no"]].sum())
        answer_probs["other"] = 1 - answer_probs["yes"] - answer_probs["no"]

        with open(f"data/results/{MODEL_DIRNAME}/results_raw_{q}.jsonl", "a") as f:
            f.write(
                json.dumps({"id": i, "prompt": row[q], "response": answer_probs}) + "\n"
            )
