from transformers import AutoTokenizer, AutoModelForCausalLM
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

# Models: "mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mixtral-8x7B-Instruct-v0.1", "mistralai/Mixtral-8x22B-Instruct-v0.1"
MODEL_NAME = "mistralai/Mixtral-8x22B-Instruct-v0.1"
MODEL_DIRNAME = MODEL_NAME.replace("/", "--")
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
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

        messages = [{"role": "user", "content": row[q]}]

        # Encode the message
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        inputs = torch.tensor(inputs, device=model.device, dtype=torch.long).reshape(
            1, -1
        )

        # Feedforward the input
        outputs = model(inputs, output_hidden_states=False)

        # The the probability of next token
        probs = torch.softmax(outputs[0][0, -1], axis=0)
        probs = probs.cpu().detach().float().numpy()

        # Handle BFloat16 errors
        probs[probs < 0.0] = 0.0
        probs = probs / probs.sum()

        # Map token ids to answers
        answer_probs = {}
        answer_probs["yes"] = float(probs[answer2tokid["yes"]].sum())
        answer_probs["no"] = float(probs[answer2tokid["no"]].sum())
        answer_probs["other"] = 1 - answer_probs["yes"] - answer_probs["no"]

        with open(f"data/results/{MODEL_DIRNAME}/results_raw_{q}.jsonl", "a") as f:
            f.write(
                json.dumps({"id": i, "prompt": row[q], "response": answer_probs}) + "\n"
            )
