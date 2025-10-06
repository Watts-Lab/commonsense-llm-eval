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

# Models: "Qwen/Qwen2-0.5B-Instruct", "Qwen/Qwen2-1.5B-Instruct",
# "Qwen/Qwen2-7B-Instruct", "Qwen/Qwen2-57B-A14B-Instruct",
# "Qwen/Qwen2-72B-Instruct"
MODEL_NAME = "Qwen/Qwen2-72B-Instruct"
MODEL_DIRNAME = MODEL_NAME.replace("/", "--")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
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

        messages = [{"role": "user", "content": row[q]}]

        # Encode the message
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

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
