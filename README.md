# Evaluating Commonsense Intelligence in LLMs

Data and code to replicate results in the following paper:

- Title: **"Empirically evaluating commonsense intelligence in large language models with large-scale human judgments"**.
- Authors: **Tuan Dung Nguyen**, **Duncan J. Watts** and **Mark E. Whiting**.
- Corresponding author: Tuan Dung Nguyen. Email: joshtn@seas.upenn.edu.

## Overview of repository

#### Set up an Anaconda environment

```bash
conda env create -f env.yml
```

#### Code

All code is in the [`src`](./src) directory. It contains:

- [IndividualCommonSense.ipynb](./src/IndividualCommonSense.ipynb)
- [GroupCommonSense.ipynb](./src/GroupCommonSense.ipynb)
- [StatementFeatureAnalysis.ipynb](./src/StatementFeatureAnalysis.ipynb): analysis of individual-level common sense with respect to subsets of statements.
- [SystemPromptAnalysis.ipynb](./src/SystemPromptAnalysis.ipynb): analysis of the effect of system prompts LLM outputs.
- [run_models](./src/run_models): scripts to run inference on LLMs.

#### Data

All data is in the [`data`](./data) directory. It contains:

- [results](./data/results): statement ratings by all LLMs and humans. The human ratings are in [individual_ratings.csv](./data/results/individual_ratings.csv) and [group_ratings.csv](./data/results/group_ratings.csv).
- [demographics.csv](./data/demographics.csv): demographics of human raters.
- [raw_statement_corpus.csv](./data/raw_statement_corpus.csv)
- [statements_and_prompts.csv](./data/statements_and_prompts.csv): statements and prompts used to query LLMs.

#### Figures

All figures in the paper are in the [`figures`](./figures) directory.

## Notes

Given a statement, we will ask LLMs questions via 3 independent prompts:

1. `Consider the statement, "[STATEMENT]." Do you agree with this statement? You must start your answer with a "yes" or "no".`
2. `Consider the statement, "[STATEMENT]." Do you think other people would agree with this statement? You must start your answer with a "yes" or "no".`
3. `Consider the statement, "[STATEMENT]." Do you think this statement is common sense? You must start your answer with a "yes" or "no".`

The prompt for each model will be in this format:

```txt
messages = [
    {"role": "user", "content": prompt}
]
```

## Handling model uncertainty in responses

Source: [Open LLM Leaderboard](https://huggingface.co/blog/evaluating-mmlu-leaderboard).

LLMs are autoregressive models, which means they "answer" these questions by sampling their vocabulary one token at a time.

Each time a token is sampled, we have a multinomial distribution over the vocabulary. In our case, the first token is sampled from something like `{"a": 0.0001, ..., "No": 0.45, ..., "Yes": 0.5, ..., "zygote": 0.0002}`. We can never be sure that a model will output a syntactically correct `Yes` or `No`, because the probability for all possible tokens is strictly positive.

So, how do we deal with this?

### Solution: only consider "legal" tokens

If we expect the model to output `Yes` or `No` only, then we consider all other tokens to be `other`, representing inconsistencies.

For example, we have observed a model outputting `As a AI language model I cannot answer...` This is likely because human alignment incentivizes the model not to answer some (potentially problematic) questions. In this case, the first token `As` is considered to be the `other` token.

Then, we will _sum_ the probabilities of all "illegal" tokens into the final probability of the `other` token. Back to the example above, if we have the following distribution over all tokens

```
{"a": 0.0001, ..., "No": 0.45, ..., "Yes": 0.5, ..., "zygote": 0.0002}
```

then the result will be

```
{"Yes": 0.5, "No": 0.45, "other": 0.05}
```

#### Nuance 1: Multiple tokens of type "Yes" or "No"

The strange thing about byte pair encoding is that tokens are not "words". As a result we would have tokens that are `Yes`, `yes`, `"Yes`, etc. which are all semantically the same.

Solution: lower-case the token. Then remove all non-alphabetic characters from it. If the result matches `yes` or `no` exactly, then the token qualifies.

For all tokens that satisfy this, we sum their probabilities. For example, if we have something like

```
{"yes": 0.4, "Yes": 0.3, '"Yes': 0.1}
```

the result will be

```
{"yes": 0.8}
```

#### Nuance 2: When we only have access to top-k tokens

OpenAI's [chat completion API](https://platform.openai.com/docs/api-reference/chat/create) only allows us access to the 5 tokens with the highest probabilities. For example

```
No 0.9998539191008537
As 5.561703604236983e-05
"No 4.7571771897529546e-05
Yes 1.593454761328504e-05
** 7.645479498605508e-06
```

We handle the 3 corresponding edges as follows.

##### Edge case 1: if the top 5 contain both "yes" and "no" tokens, then we rescale the top 5 probabilities so that they add up to 1.

For example, if we have

```
{"yes": 0.7, "Yes": 0.1, "no": 0.05, "**": 0.01, "I": 0.01}
```

this will be transformed to

```
{"yes": 0.8, "no" 0.05, "other": 0.02}
```

and then finally to

```
{"yes": 0.91954023, "no": 0.05747126, "other": 0.02298851}
```

now the probabilities of all options add up to 1 and we're done.

##### Edge case 2: if the top 5 only contain only `yes` tokens but not `no` tokens, then we consider the rest of the probability mass (`1 - sum(top 5 probs)`) to be the probability for `no` tokens.

For example, if we have

```
{"Yes": 0.8, '"Yes': 0.1, "I": 0.04, "As": 0.03, "**": 0.02}
```

this will be transformed to

```
{"yes": 0.9, "other": 0.08}
```

The total probability mass is `0.9 + 0.08 = 0.98`, which means the rest of the mass is `1 - 0.98 = 0.02`, which will be assigned to the missing token, `no`. Therefore, we have

```
{"yes": 0.9, "no": 0.02, "other: 0.08}
```

This applies in the same way as when `yes` is missing and `no` is present.

##### Edge case 3: if the top 5 contains only `other` tokens. This happens much less frequently but is possible. In this case we take the rest of the probability mass and divide it equally to `yes` and `no`.

For example, if we have

```
{"As": 0.9, "I": 0.05, "**": 0.03, ",": 0.01, "<": 0.005}
```

then this will be transformed to

```
{"other": 0.995}
```

The rest of the mass is `1 - 0.995 = 0.005`, which will be divided equally to `yes` and `no` answers. Therefore, we finally have

```
{"yes": 0.0025, "no": 0.0025, "other": 0.995}
```

### Alternative solution

Another way to handle a model's uncertainty with respect to its sampling distribution is to _sum over the log probabilities of its response sequence_.

Consider the following response by a model: `Yes, I agree with the statement.` If we have access to the probability of each output token (`p("Yes")`, `p(",")`, `p(I)` and so on), then we can consider the log probability of the model's answer to be:

```
logp(answer) = logp("Yes") + logp(",") + logp(I) + ...
```

Sometimes this log probability is normalized by the sequence length, giving us the log perplexity. Then, exponentiating the log perplexity will give us the probability of the sequence, and this is treated as the probability of the answer:

```
p(answer) = exp(logp(answer) / num_tokens)
```

This method is typically used in multiple-choice Q&A settings, where a model is asked to choose one option out of A, B, C an D. The metric is used by [Eleuther AI's Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/e47e01beea79cfe87421e2dac49e64d499c240b4/lm_eval/models/gpt3.py#L25) framework and the [TruthfulQA benchmark](https://github.com/sylinrl/TruthfulQA/blob/fdd8ad1c0d00a478cf8b0bb41a3ad8378c16293b/truthfulqa/models.py#L121).

This method does _not_ give a straightforward account of probability, is used primarily in multiple-choice questions settings (which is not very relevant to our yes/no setting), and requires the language model to generate until the end of the sequence. Therefore, we do not choose this alternative solution.

### Other problems

#### GPT-4 Reproducibility

OpenAI offers [reproducibility](https://platform.openai.com/docs/guides/text-generation/reproducible-outputs) by access to the `seed` parameter in calling the text completion API.

However, `seed` is not enough to ensure determinism. In the response object, the `system_fingerprint` refers to the identifier for the current combination of model weights, infrastructure, and other configuration options used by OpenAI servers to generate the completion.

According to OpenAI's [cookbook](https://cookbook.openai.com/examples/reproducible_outputs_with_the_seed_parameter), if the `seed`, request parameters (e.g., the prompt, `top_p`, `temperature`), and `system_fingerprint` all match across your requests, then model outputs will **mostly be identical**. There is a small chance that responses differ even when request parameters and system_fingerprint match, due to the inherent non-determinism of our models.

#### GPT-5

This model is a reasoning LLM by default. We cannot turn this behavior off, so we discourage it by adding the instruction `Do not include anything else, such as an explanation or reasoning` to the prompt, and use `reasoning_effort="minimal"` in the API call.

## Notes on running LLMs locally

### LLaMA-2

The first token that LLaMA-2-chat is actually not `Yes` or `No`, but a special token `"_"` or which stands for `SPIECE_UNDERLINE`.

- See [this issue on GitHub](https://github.com/huggingface/transformers/issues/26273).
- This is because LLaMA-2 was trained to predict this `SPIECE_UNDERLINE` token before starting an actual response.
- **Solution**: append the `SPIECE_UNDERLINE` token to a prompt, so that the first token that LLaMA-2 is supposed to generate will be a definitive `yes` or `no`.
