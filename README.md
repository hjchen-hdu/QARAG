# QARAG

`QARAG` is a working fork scaffolded from `SubgraphRAG`. The current goal is not to change behavior yet, but to make the original code easier to extend toward:

```text
Query-aware weak supervision
+ flow-guided triple features
+ bottleneck-style compact evidence retrieval
+ KGQA-oriented LLM prompting
```

This directory intentionally contains the lightweight code skeleton only. Large experiment artifacts from `SubgraphRAG` were not copied:

- `retrieve/data_files/*/processed`
- `retrieve/data_files/*/emb`
- `retrieve/data_files/*/triple_scores`
- `reason/results`
- `reason/gpt_labeled`
- `reason/scored_triples`

## Structure

```text
QARAG/
├── retrieve/
│   ├── emb.py
│   ├── train.py
│   ├── inference.py
│   ├── eval.py
│   ├── configs/
│   ├── requirements/
│   └── src/
│       ├── dataset/
│       ├── model/
│       └── config/
├── reason/
│   ├── main.py
│   ├── prompts.py
│   ├── preprocess/
│   └── metrics/
└── plan/
```

## Current Code Logic

The migrated baseline still follows `SubgraphRAG`:

```text
1. emb.py
   Precompute question/entity/relation embeddings.

2. src/dataset/retriever.py
   Build weak labels from shortest paths between topic entities and answer entities.

3. src/model/retriever.py
   Score each triple with question, head, relation, tail, topic PE, and DDE features.

4. train.py
   Train with binary cross entropy against shortest-path triple labels.

5. inference.py
   Rank triples and save top-K scored triples.

6. reason/
   Convert retrieved triples into LLM prompts and evaluate KGQA answers.
```

## Planned QARAG Extension Points

The code now contains `QARAG TODO` comments in the exact places where the new modules should be inserted.

Recommended order:

1. **Query-aware weak label builder**  
   Add this in `retrieve/src/dataset/retriever.py`. It should replace binary shortest-path labels with soft labels from relation/path-query similarity.

2. **Flow prior computer**  
   Add this after sample graph construction in `retrieve/src/dataset/retriever.py` or as a separate utility module. It should output node/edge flow features for each triple.

3. **Retriever feature augmentation**  
   Add relation-query similarity, triple-query similarity, and flow prior features in `retrieve/src/model/retriever.py`.

4. **Bottleneck regularization**  
   Add the KL/sparsity terms in `retrieve/train.py`, next to the original BCE loss.

5. **Adaptive evidence selection**  
   Replace fixed top-K-only output in `retrieve/inference.py` with threshold/budget-based subgraph selection.

6. **Evidence prompt serialization**  
   Extend `reason/preprocess/prepare_prompts.py` to support compact subgraph plus optional path-style serialization.

## Original Source

This scaffold is based on:

```tex
@inproceedings{li2024subgraphrag,
    title={Simple is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation},
    author={Li, Mufei and Miao, Siqi and Li, Pan},
    booktitle={International Conference on Learning Representations},
    year={2025}
}
```
