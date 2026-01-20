from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import json
import pandas as pd
import os
import jsonlines
from tqdm import tqdm
import torch

# Load statements
statements = pd.read_csv("data/statements_and_prompts.csv")

# Models: "tiiuae/falcon-180B-chat", "tiiuae/falcon-40b-instruct", "tiiuae/falcon-7b-instruct"
MODEL_NAME = "tiiuae/falcon-180B-chat"
MODEL_DIRNAME = MODEL_NAME.replace("/", "--")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
)
model.eval()

# Load token to answer mapping
with open(f"data/results/{MODEL_DIRNAME}/answer2tokid.json", "r") as f:
    answer2tokid = json.load(f)

questions_unanswered = {q: set(range(len(statements))) for q in ["q1", "q2", "q3"]}
for q in ["q1", "q2", "q3"]:
    if os.path.exists(f"data/results/{MODEL_DIRNAME}/results_raw_{q}.jsonl"):
        with open(f"data/results/{MODEL_DIRNAME}/results_raw_{q}.jsonl", "r") as fp:
            for line in jsonlines.Reader(fp):
                questions_unanswered[q].remove(line["id"])

FALCON_SUFFIX = [193, 49, 268, 1043, 37]
for q in ["q1", "q2", "q3"]:
    for i, row in tqdm(statements.iterrows(), total=len(statements)):
        if i not in questions_unanswered[q]:
            continue

        # Encode the message
        prompt = row[q]
        prompt = f"User: {prompt}"
        inputs = tokenizer(prompt)["input_ids"]

        # Add Falcon-specific suffix to the input
        inputs.extend(FALCON_SUFFIX)

        # Feedforward the input
        inputs = torch.tensor(inputs, device=model.device, dtype=torch.long).reshape(
            1, -1
        )
        outputs = model(inputs, output_hidden_states=False)

        # The the probability of next token
        probs = torch.softmax(outputs[0][0, -1], axis=0)
        probs = probs.cpu().detach().numpy()

        # Map token ids to answers
        answer_probs = {}
        answer_probs["yes"] = float(probs[answer2tokid["yes"]].sum())
        answer_probs["no"] = float(probs[answer2tokid["no"]].sum())
        answer_probs["other"] = 1 - answer_probs["yes"] - answer_probs["no"]

        with open(f"data/results/{MODEL_DIRNAME}/results_raw_{q}.jsonl", "a") as f:
            f.write(
                json.dumps({"id": i, "prompt": row[q], "response": answer_probs}) + "\n"
            )
