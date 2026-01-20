# Evaluating Commonsense Knowledge in Large Language Models

Data and code to replicate results in the following paper:

- Title: **"A large-scale evaluation of commonsense knowledge in humans and large language models"**.
- Authors: **Tuan Dung Nguyen**, **Duncan J. Watts** and **Mark E. Whiting**.
- Corresponding author: Tuan Dung Nguyen. Email: joshtn@seas.upenn.edu.
- Preprint URL: https://arxiv.org/abs/2505.10309.

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
