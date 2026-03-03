# Evaluating Commonsense Knowledge in Large Language Models

Data and code to reproduce results in the following paper.

**Tuan Dung Nguyen, Duncan J. Watts, and Mark E. Whiting, A large-scale evaluation of commonsense knowledge in humans and large language models. _PNAS Nexus_ 5(3): pgag029. 2026. https://doi.org/10.1093/pnasnexus/pgag029.**

> **Abstract**—Commonsense knowledge, a major constituent of AI, is primarily evaluated in practice by human-prescribed ground-truth labels. An important, albeit implicit, assumption of these labels is that they accurately capture what any human would think, effectively treating human common sense as homogeneous. However, recent empirical work has shown that humans vary enormously in what they consider commonsensical; thus what appears self-evident to one benchmark designer may not be so to another. Here, we propose a method for assessing commonsense knowledge in AI, specifically in large language models (LLMs) that incorporates empirically observed heterogeneity among humans by measuring the correspondence between a model’s judgment and that of a human population. We first find that, when treated as independent survey respondents, most LLMs remain below the human median in their individual commonsense competence. Second, when used as simulators of a hypothetical population, LLMs correlate with real humans only modestly in the extent to which they agree on the same set of statements. In both cases, smaller, open-weight models are surprisingly more competitive than larger, proprietary frontier models. Our evaluation framework, which ties commonsense knowledge to its cultural basis, contributes to the growing call for adapting AI models to human collectivities that possess different, often incompatible, social stocks of knowledge.

A preprint is also available at https://arxiv.org/abs/2505.10309.

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

## Use and Citation

This article is distributed under the Creative Commons Attribution License (CC BY 4.0). You are free to reuse, distribute and reproduce its content in any medium, provided that the original work is properly cited.

Please cite the article using the following BibTeX entry.

```
@article{nguyenLargescaleEvaluationCommonsense2026,
    title = {A large-scale evaluation of commonsense knowledge in humans and large language models},
    author = {Nguyen, Tuan Dung and Watts, Duncan J. and Whiting, Mark E.},
    year = 2026,
    journal = {PNAS Nexus},
    volume = {5},
    number = {3},
    pages = {pgag029},
    publisher = {Oxford University Press},
    doi = {10.1093/pnasnexus/pgag029},
}
```
