# QARAG Migration Notes

生成日期：2026-05-21

## 迁移范围

本目录从 `Experiments/SubgraphRAG` 迁移了代码骨架：

- `retrieve/`：embedding、retriever training、inference、retrieval eval、configs、requirements、`src/`
- `reason/`：LLM runner、prompts、preprocess、metrics、configs
- 顶层 `README.md` 与 `LICENSE`

未迁移大体积运行产物：

- processed datasets
- embedding caches
- triple score caches
- scored triples
- LLM results
- GPT-labeled files
- `.git` metadata

## 当前代码逻辑

当前 `QARAG` 仍然等价于 SubgraphRAG baseline：

```text
emb.py
-> src/dataset/retriever.py builds shortest-path triple labels
-> src/model/retriever.py scores triples with MLP + DDE
-> train.py optimizes BCE
-> inference.py emits top-K scored triples
-> reason/ serializes triples into LLM prompts
```

## 已标注的扩展插入点

代码中已经加入 `QARAG TODO` 注释。主要位置如下：

| 模块 | 文件 | 插入点 |
|---|---|---|
| query-aware weak labels | `retrieve/src/dataset/retriever.py` | `_get_triple_scores`, `_extract_paths_and_score`, `_score_triples` |
| flow prior features | `retrieve/src/dataset/retriever.py` | `_assembly` after loading embeddings |
| retriever feature expansion | `retrieve/src/model/retriever.py` | `pred_in_size` and `h_triple` construction |
| bottleneck objective | `retrieve/train.py` | after baseline BCE |
| adaptive evidence selection | `retrieve/inference.py` | after `pred_triple_scores` |
| connectivity/path repair | `retrieve/inference.py` | after selected triples are built |
| retrieval metrics | `retrieve/eval.py` | metric construction |
| prompt serialization | `reason/preprocess/prepare_prompts.py` | scored mode branch |
| reasoner contract | `reason/main.py` | input/output boundary |

## 建议实现顺序

1. 保持 SubgraphRAG baseline 可运行。
2. 在 `retriever.py` 数据侧加入 near-shortest candidate path collection。
3. 用 relation/path embedding similarity 生成 soft labels。
4. 训练 loss 第一版保持 BCE，不加入额外 ranking、sparse、budget 或 bottleneck loss。
5. 在 model 侧加入 query-aware semantic encoder，与 DDE-style structure encoder 并行。
6. 在 `inference.py` 保持 fixed top-K scored triples，先不做 adaptive evidence selection。
7. reason 部分暂不修改，等 retriever 结果稳定后再决定是否调整 prompt serialization。

## 设计边界

第一版不要引入：

- LLM rational path labeling；
- autoregressive path decoder；
- full QAFD passage-entity graph；
- GNN-RAG answer-node GNN；
- heavy graph attention / multi-router retriever；
- bottleneck KL、size penalty 或其他没有明确 selection variable 的正则项；
- reason prompt 改造。

第一版只需要在 SubgraphRAG 主干上加：

```text
query-aware soft weak labels
+ lightweight parallel dual-encoder retriever
+ BCE-based triple relevance learning
```

## Retriever Model 设计：并行双编码器

更新日期：2026-05-27

实验复现与 ablation 细节见 `qarag_experiment_plan.md`。

当前 QARAG 第一版只关注 retriever。输入可以抽象为：

```text
query q + candidate subgraph G_q = (V_q, E_q)
```

目标是在 `G_q` 中给每个 candidate triple `tau_i = (h_i, r_i, t_i)` 输出 relevance score：

```text
s_i = Retriever(q, G_q, tau_i)
```

训练监督仍来自 query-aware weak labels，loss 第一版只保留 BCE：

```text
L = BCEWithLogits(s_i, y_i)
```

其中 `y_i` 来自 shortest / near-shortest path pool 的 query-aware 筛选结果。暂时不加入 `L_sparse`、`L_budget`、pairwise ranking loss 或 bottleneck KL，避免方法变重且动机不清。

### 设计动机

参考 GraphMixer / *Do We Really Need Complicated Model Architectures for Temporal Networks?* 的拆分思想：不直接堆复杂 GNN、attention 或 path decoder，而是把 retriever 分成两个轻量分支：

```text
                 +-> Structure Encoder ------------ z_struct(tau_i)
query q, G_q ----|
                 +-> Query-Aware Semantic Encoder -- z_sem(tau_i, q)

s_i = MLP_score([z_struct(tau_i), z_sem(tau_i, q)])
```

这里不是直接复用 temporal network 任务，而是借鉴其“简单编码器 + shallow classifier”的模型哲学。

### Structure Encoder

Structure Encoder 负责回答：

```text
这个 triple 在当前 subgraph 结构上是否像 evidence？
```

第一版直接复用 SubgraphRAG 中的 DDE-style propagation，不额外引入 heavy GNN：

```text
z_struct(tau_i) = MLP_struct([
  DDE(h_i),
  DDE(t_i),
  topic_pe(h_i),
  topic_pe(t_i)
])
```

可用结构信号包括：

- topic entity indicator；
- forward / reverse DDE propagation feature；
- head / tail 相对 topic entity 的结构位置；
- head / tail 是否靠近 query entry；
- 可选：degree、path position、是否出现在 weak path pool 中。

第一版建议先只使用已有 DDE 和 topic PE，避免结构分支过度扩展。

### Query-Aware Semantic Encoder

Query-Aware Semantic Encoder 负责回答：

```text
这个 triple 的语义是否和 query 匹配？
```

它和 Structure Encoder 并行，不依赖 DDE 输出。第一版可以基于已有 embedding 构造 query-triple matching feature：

```text
z_sem(tau_i, q) = MLP_sem([
  q_emb,
  h_emb,
  r_emb,
  t_emb,
  q_emb * r_emb,
  cos(q_emb, r_emb),
  cos(q_emb, triple_emb)
])
```

其中最重要的是 relation-query matching：

```text
relation_query_sim = cos(q_emb, r_emb)
```

因为 KGQA 中 query 的核心语义通常首先对应 relation，例如 `Who directed Inception?` 与 `film.director` / `directed_by` 的匹配。

### Fusion Scorer

两个 encoder 并行输出后，用 late fusion 得到 triple score：

```text
h_i = concat(z_struct(tau_i), z_sem(tau_i, q))
s_i = MLP_score(h_i)
```

第一版不要加 gate。直接 concat + MLP 更容易复现，也更方便做 ablation。

### 推荐 Ablation

后续实验按从简单到完整的顺序做：

1. `DDE only`：原 SubgraphRAG retriever baseline。
2. `Semantic only`：只用 query-aware semantic matching。
3. `DDE + Semantic`：并行双编码器。
4. `DDE + Semantic + query-aware weak labels`：完整 QARAG retriever。

核心判断指标不应只看 BCE loss，还应看 retrieval 结果：

- `Recall@100/300/500`
- `Precision@100/300/500`
- `NDCG@100/300/500`
- `Answer Coverage@100/300/500`
- `Path Coverage@100/300/500`

### 代码插入位置

后续复现时建议只改 retriever 侧：

- `retrieve/src/model/retriever.py`
  - 在 `Retriever.__init__` 中增加 semantic feature 维度；
  - 在 `Retriever.forward` 中计算 `relation_query_sim`、`triple_query_sim`；
  - 在构造 `h_triple` 时拼接 structure feature 和 semantic feature；
  - 保持最终 `self.pred` 为 shallow MLP。
- `retrieve/src/dataset/retriever.py`
  - 如果 semantic feature 需要预计算，可以在 `_assembly` 中准备 per-triple feature；
  - 第一版更建议直接在 model forward 中基于 `q_emb`、`entity_embs`、`relation_embs` 计算，减少缓存复杂度。
- `retrieve/train.py`
  - loss 先保持 BCE / confidence-weighted BCE 二选一；
  - 不加入额外 regularization。
- `reason/`
  - 暂不修改。
