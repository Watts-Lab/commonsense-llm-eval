import os
from typing import Dict
from collections import defaultdict
from math import exp
import json
import pandas as pd
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def _alpha(char):
    return char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def token_to_answer(token: str) -> str:
    """
    Convert a token to a `yes`, `no` or `other` answer.

    Args:
    token (str): token to convert
    """
    # Lowercase the token
    token = token.lower()
    # Remove non-alphabetic characters
    token = "".join(filter(_alpha, token))
    # Now do exact matching
    if token == "yes":
        return "yes"
    elif token == "no":
        return "no"
    else:
        return "other"


def process_gpt_answers(token_probs: Dict, is_log_prob=True) -> Dict:
    """
    Process the GPT-3 outputs to extract the generated text.

    Args:
    token_probs (Dict): log probabilities of the top-k
        tokens in the generated text.
    is_log_prob (bool): whether the probabilities are
        (natural) log probabilities.

    Returns:
    results: Dict with the probabilities of each answer,
        "yes", "no" and "other".
    """
    results = defaultdict(float)
    # Step 1: map tokens to possible answers
    for token, prob in token_probs.items():
        answer = token_to_answer(token)
        if is_log_prob:
            # This is log probability, so we need to exponentiate
            results[answer] += exp(prob)
        else:
            # This is aleady a probability, so no need to exponentiate
            # Check if prob is negative (because of fp errors)
            if prob < 0:
                prob = 0
            results[answer] += prob

    # Step 2: Check for missing answers in the top k
    if "yes" not in results and "no" not in results:
        #
        remaining_prob = 1 - sum(results.values())
        results["yes"] = remaining_prob / 2
        results["no"] = remaining_prob / 2
    elif "no" not in results:
        remaining_prob = 1 - sum(results.values())
        results["no"] = remaining_prob
    elif "yes" not in results:
        remaining_prob = 1 - sum(results.values())
        results["yes"] = remaining_prob

    # Step 3: Scale so that the probabilities sum to 1
    total = sum(results.values())
    for answer in results:
        results[answer] /= total

    # Step 4: Check if "other" is in the results
    if "other" not in results:
        results["other"] = 0.0

    return results


def create_tokid_to_answer(hf_model_name: str):
    """
    Take a Hugging Face tokenizer and return a dictionary mapping
    "yes" and "no" answers to compatible token ids in the vocabulary.

    Args:
    hf_model_name (str): name of the Hugging Face model.
    """
    # Load tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    # Go over all tokens and map them to "yes"/"no" answers
    # "other" answers are simply the rest
    answer2tokid = {"yes": [], "no": []}
    for token, tokid in tokenizer.get_vocab().items():
        answer = token_to_answer(token)
        if answer in answer2tokid:
            # print("Without REGEX:", answer, token, tokid)
            answer2tokid[answer].append(tokid)

    return answer2tokid


def load_results_hf(model_name: str, verbose: bool = False):
    """
    Load the common sense annotation results by
    a Hugging Face model.

    Args:
    model_name (str): name of the Hugging Face model.
    verbose (bool): whether to print messages.

    Returns:
    results: Dict with answers to each question, in probabilities
        of "yes", "no" and "other".
    """

    # TODO: Check this if human annotations change
    n_statements = 4407

    # Replace / with -- to avoid path issues
    model_name = model_name.replace("/", "--")

    base_path = os.path.join("data", "results", model_name)

    results = {}
    for q in ["q1", "q2", "q3"]:
        path = os.path.join(base_path, f"results_raw_{q}.jsonl")
        annots = pd.DataFrame(
            0.0, index=range(n_statements), columns=["yes", "no", "other"]
        )
        with open(path, "r") as fp:

            # Go over each line
            for _, line in tqdm(
                enumerate(fp), leave=False, desc="Statement", disable=not verbose
            ):
                obj = json.loads(line)

                statement_id = obj["id"]

                # Extract yes/no/other answers with their associated (log) probs
                answers = obj["response"]

                # Map answers
                answers = process_gpt_answers(token_probs=answers, is_log_prob=False)

                # Add to data frame
                for answer, prob in answers.items():
                    annots.loc[statement_id, answer] = prob

        results[q] = annots
    return results


def raw_str_to_binary_answer(answer: str):
    """
    Process the raw string answer from the model into
    one of three possible answers: "yes", "no" or "other".
    """

    # Lowercase the answer
    answer = answer.lower()

    # Remove non-alphabetic characters at the beginning (if any)
    while len(answer) > 0 and not answer[0].isalpha():
        answer = answer[1:]

    if answer.startswith("yes"):
        return "yes"
    elif answer.startswith("no"):
        return "no"
    else:
        return "other"


def load_results_freq(model_name: str, verbose: bool = False):
    """
    Load the common sense annotation results by proprietary
    models, where we get probabilities via repeated
    generations given the same prompt.

    Args:
    model_name (str): name of the model.
    verbose (bool): whether to print messages.

    Returns:
    results: Dict with answers to each question, in probabilities
        of "yes", "no" and "other".
    """

    # TODO: Check this if human annotations change
    n_statements = 4407

    base_path = os.path.join("data", "results", model_name)
    results = {}
    for q in ["q1", "q2", "q3"]:
        annots = pd.DataFrame(
            0.0, index=range(n_statements), columns=["yes", "no", "other"]
        )
        for it in range(1, 1000, 1):
            # Set path to results
            if it == 1:
                path = os.path.join(base_path, f"results_raw_{q}.jsonl")
            else:
                path = os.path.join(base_path, f"results_raw_{q}_trial{it}.jsonl")

            # Done if this iteration doesn't exist
            if not os.path.exists(path):
                if verbose:
                    print(f"Question {q} has {it - 1} repetitions")
                break

            with open(path, "r") as fp:
                for _, line in tqdm(
                    enumerate(fp), leave=False, desc="Statement", disable=not verbose
                ):
                    obj = json.loads(line)

                    statement_id = obj["id"]

                    # Extract yes/no/other answers with their associated (log) probs
                    answers = obj["response"]
                    answers = raw_str_to_binary_answer(answers)

                    assert answers in annots.columns
                    annots.loc[statement_id, answers] += 1

        # Normalize to get frequencies of each answer
        annots = annots.div(annots.sum(axis=1), axis=0)

        results[q] = annots

    return results


def load_annotations_gpt(model_name: str, trial_no: int = 1, verbose: bool = False):
    """
    Load the common sense annotation results by GPT-3 and GPT-4.

    Args:
    model_name (str): name of the model: "gpt3.5" or "gpt4".
    trial_no (int): trial number.
    verbose (bool): whether to print messages.

    Returns:
    results: Dict with answers to each question, in probabilities
        of "yes", "no" and "other".
    """

    # TODO: Check this if human annotations change
    n_statements = 4407

    base_path = os.path.join("data", "results", model_name)
    results = {}

    for q in ["q1", "q2", "q3"]:
        # Set path to results
        if trial_no == 1:
            path = os.path.join(base_path, f"results_raw_{q}.jsonl")
        else:
            path = os.path.join(base_path, f"results_raw_{q}_trial{trial_no}.jsonl")

        annots = pd.DataFrame(
            0.0, index=range(n_statements), columns=["yes", "no", "other"]
        )

        with open(path, "r") as fp:
            for _, line in tqdm(
                enumerate(fp), leave=False, desc="Statement", disable=not verbose
            ):
                obj = json.loads(line)

                statement_id = obj["id"]

                # Extract yes/no/other answers with their associated (log) probs
                answers = {}
                for tok_logprob in obj["response"]["choices"][0]["logprobs"]["content"][
                    0
                ]["top_logprobs"]:
                    answers[tok_logprob["token"]] = tok_logprob["logprob"]

                # Map answers
                answers = process_gpt_answers(token_probs=answers, is_log_prob=True)

                # Add to data frame
                for answer, prob in answers.items():
                    annots.loc[statement_id, answer] = prob

        results[q] = annots

    return results


def load_annotations_gpt_from_batch(
    model_name: str, trial_no: int = 1, verbose: bool = False, questions=["q1", "q2"]
):
    """
    Load the common sense annotation results by GPT-3 and GPT-4.

    Args:
    model_name (str): name of the model: "gpt3.5" or "gpt4".
    trial_no (int): trial number.
    verbose (bool): whether to print messages.

    Returns:
    results: Dict with answers to each question, in probabilities
        of "yes", "no" and "other".
    """

    # TODO: Check this if human annotations change
    n_statements = 4407

    base_path = os.path.join("data", "results", model_name)
    results = {}

    for q in questions:
        # Set path to results
        if trial_no == 1:
            path = os.path.join(base_path, f"results_raw_{q}.jsonl")
        else:
            path = os.path.join(base_path, f"results_raw_{q}_trial{trial_no}.jsonl")

        annots = pd.DataFrame(
            0.0, index=range(n_statements), columns=["yes", "no", "other"]
        )

        with open(path, "r") as fp:
            for statement_id, line in tqdm(
                enumerate(fp), leave=False, desc="Statement", disable=not verbose
            ):
                obj = json.loads(line)

                # statement_0_question_q1
                custom_id = obj["custom_id"]
                s_id = int(custom_id.split("_")[1])
                assert s_id == statement_id

                # Extract yes/no/other answers with their associated (log) probs
                answers = {}
                for tok_logprob in obj["response"]["body"]["choices"][0]["logprobs"][
                    "content"
                ][0]["top_logprobs"]:
                    answers[tok_logprob["token"]] = tok_logprob["logprob"]

                # Map answers
                answers = process_gpt_answers(token_probs=answers, is_log_prob=True)

                # Add to data frame
                for answer, prob in answers.items():
                    annots.loc[statement_id, answer] = prob

        results[q] = annots

    return results


def preprocess_raw_hf_cot_answers(
    path_to_raw: str,
    model_name: str,
    statements_path: str = os.path.join("data", "statements_and_prompts_cot.csv"),
    verbose: bool = False,
):
    """
    Preprocess the raw chain-of-thought answers from a Hugging Face model.
    The raw answers are stored in JSONL files (e.g., `results_raw_q1_raw.jsonl`),
    where each line is an object of the form [{"generated_text": "question_answer"}].
    The processed answers are stored in JSONL files (e.g., `results_raw_q1_raw.jsonl`),
    where each line is a JSON object of the form
    {"id": statement_id, "prompt": "question", "response": "answer"}.

    Args:
    path_to_raw (str): path to the raw data directory.
    model_name (str): name of the Hugging Face model.
    statements_path (str): path to the statements and prompts file.
        Default is "data/statements_and_prompts_cot.csv".
    verbose (bool): whether to print messages.
    """

    import ast
    from transformers import AutoTokenizer

    BASE_PATH = path_to_raw
    statements = pd.read_csv(statements_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for q in ["q1", "q2", "q3"]:
        raw_data = []
        data = []

        with open(os.path.join(BASE_PATH, f"results_raw_{q}_raw.jsonl"), "r") as fp:
            for _, line in tqdm(
                enumerate(fp), leave=False, desc="Statement", disable=not verbose
            ):
                raw_data.append(ast.literal_eval(line)[0]["generated_text"])

        for i, (question, question_and_answer) in enumerate(
            zip(statements[q], raw_data)
        ):
            q_template = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
            assert question_and_answer.startswith(q_template)
            s = question_and_answer[len(q_template) :]
            data.append({"id": i, "prompt": question, "response": s})

            with open(os.path.join(BASE_PATH, f"results_raw_{q}.jsonl"), "a") as fp:
                fp.write(
                    json.dumps({"id": i, "prompt": question, "response": s}) + "\n"
                )


def load_results_gpt5(model_name: str = "gpt-5-2025-08-07", verbose: bool = False):
    """
    Load the common sense annotation results by GPT-5. This is a reasoning model,
    and its answers were generated via batched API calls. Each question was asked
    6 times, and there were 8 choices (parameter "n" in OpenAI API) per request,
    so in total we have 6 x 8 = 48 responses per question.

    Args:
    model_name (str): name of the model.
    verbose (bool): whether to print messages.

    Returns:
    results: Dict with answers to each question, in probabilities
        of "yes", "no" and "other".
    """

    # TODO: Check this if human annotations change
    n_statements = 4407

    base_path = os.path.join("data", "results", model_name)
    results = {}
    # Question 3 was not asked at all
    for q in ["q1", "q2"]:
        annots = pd.DataFrame(
            0.0, index=range(n_statements), columns=["yes", "no", "other"]
        )

        path = os.path.join(base_path, f"results_raw_{q}.jsonl")
        with open(path, "r") as fp:
            for _, line in tqdm(
                enumerate(fp), leave=False, desc="Statement", disable=not verbose
            ):
                obj = json.loads(line)

                # Format of custom_id: statement_0_question_q2_request_1
                custom_id = obj["custom_id"].split("_")
                statement_id = int(custom_id[1])
                request_id = int(custom_id[5])

                for choice in obj["response"]["body"]["choices"]:
                    answer_raw = choice["message"]["content"]
                    answer = raw_str_to_binary_answer(answer_raw)

                    assert answer in annots.columns
                    annots.loc[statement_id, answer] += 1

        annots = annots.div(annots.sum(axis=1), axis=0)

        results[q] = annots

    return results
