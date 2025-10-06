import pandas as pd
import json
from tqdm import tqdm
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai
import os
import time
import jsonlines

GOOGLE_API_KEY = os.getenv("GOOGLE_GEMINI_API")
if GOOGLE_API_KEY is None:
    GOOGLE_API_KEY = input("Input your Google AI API key:")

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel(
    model_name="models/gemini-pro",
)

statements = pd.read_csv("data/statements_and_prompts.csv")

questions_unanswered = {q: set(range(len(statements))) for q in ["q1", "q2", "q3"]}
for q in ["q1", "q2", "q3"]:
    if os.path.exists(f"data/results/gemini-pro/results_raw_{q}_trial50.jsonl"):
        with open(f"data/results/gemini-pro/results_raw_{q}_trial50.jsonl", "r") as fp:
            for line in jsonlines.Reader(fp):
                questions_unanswered[q].remove(line["id"])

for i, row in tqdm(statements.iterrows(), total=len(statements)):

    for q in ["q1", "q2", "q3"]:

        if i not in questions_unanswered[q]:
            continue

        config = genai.types.GenerationConfig(
            temperature=1,
        )

        safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }

        prompt = row[q]

        messages = [{"role": "user", "parts": [prompt]}]

        try:
            response = model.generate_content(
                prompt, generation_config=config, safety_settings=safety_settings
            )
            text = response.text
        except Exception as e:
            time.sleep(60)
            response = model.generate_content(
                prompt, generation_config=config, safety_settings=safety_settings
            )

        with open(f"data/results/gemini-pro/results_raw_{q}_trial50.jsonl", "a") as f:
            f.write(
                json.dumps({"id": i, "prompt": prompt, "response": response.text})
                + "\n"
            )
