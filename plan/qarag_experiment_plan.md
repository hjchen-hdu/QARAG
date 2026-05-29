# QARAG Experiment Plan

生成日期：2026-05-27

## 目标

QARAG 第一版只验证 retriever 改动，不改 reasoner。核心 claim 是：

```text
query-aware weak supervision
+ lightweight parallel Structure/Semantic retriever
=> better query-relevant triple retrieval under the same KGQA setting
```

因此实验优先回答三个问题：

1. QARAG 是否比 SubgraphRAG 检索到更多 answer-relevant / evidence-relevant triples？
2. query-aware weak labels 和 query-aware semantic encoder 分别带来多少收益？
3. retrieval 改进是否能在固定 reasoner 下转化为 KGQA 表现提升？

## 实验资产与统一设置

### 数据集

主实验：

- `WebQSP`
- `CWQ`

可选子集实验：

- `WebQSP-sub`
- `CWQ-sub`

第一版先使用 SubgraphRAG 相同 processed data 和 entity/relation embeddings，不重新构造 KGQA 数据。

### 固定设置

为了让比较只反映 retriever 差异，除被 ablate 的部分外保持一致：

- text encoder：`gte-large-en-v1.5`
- random seed：`42`
- retriever optimizer / lr / epoch / patience：沿用 SubgraphRAG config
- inference budget：默认保存 top-500 scored triples
- reasoner：暂不改 prompt 和 LLM

### 输出 artifacts

每个 retriever run 至少保存：

```text
retrieve/{dataset}_{run_name}/cpt.pth
retrieve/{dataset}_{run_name}/retrieval_result.pth
retrieve/{dataset}_{run_name}/metrics.json
retrieve/{dataset}_{run_name}/config.json
```

如果当前代码还没有 `metrics.json`，先用 markdown / csv 记录也可以，但最终建议补机器可读结果，方便画图。

## 实验 1：Main Retrieval Results

### 目的

验证 QARAG retriever 是否比 SubgraphRAG baseline 找回更多 query-relevant evidence。

### 方法对比

至少比较：

| 方法 | 说明 |
|---|---|
| `SubgraphRAG` | 原始 DDE + shortest-path hard label |
| `QARAG` | query-aware weak labels + parallel dual-encoder retriever |

可选比较：

| 方法 | 说明 |
|---|---|
| `Cosine` | query embedding 与 triple/relation embedding 相似度排序 |
| `Random` | 随机 triples |
| `RoG` | 如果本地有 RoG paths / scored triples，可作为外部 baseline |

### 指标

在 `K = 50, 100, 200, 400, 500` 上报告：

```text
ans_recall@K
shortest_path_triple_recall@K
gpt_triple_recall@K
```

其中：

- `ans_recall@K`：top-K triples 的 head/tail entity 是否覆盖答案实体；
- `shortest_path_triple_recall@K`：top-K triples 是否覆盖弱监督 shortest / selected path triples；
- `gpt_triple_recall@K`：top-K triples 是否覆盖 GPT-labeled relevant triples。

### 实施细节

当前 SubgraphRAG/QARAG 的 `retrieve/eval.py` 已经支持：

```bash
cd Experiments/QARAG/retrieve
python eval.py -d webqsp -p <run_dir>/retrieval_result.pth --k_list 50,100,200,400,500
python eval.py -d cwq   -p <run_dir>/retrieval_result.pth --k_list 50,100,200,400,500
```

注意：

- `gpt_triple_recall@K` 依赖 `data_files/{dataset}/gpt_triples.pth`；
- 如果某个 sample 没有 GPT-labeled triples，则不要计入该 metric 的平均；
- QARAG 的 `target_relevant_triples` 如果来自 soft label，需要明确是 `y_i > 0` 的 triples。

## 实验 2：Model Ablation

### 目的

证明模型结构收益来自 query-aware semantic encoder 与 DDE-style structure encoder 的互补，而不是简单参数量增加。

### Ablation 设置

| 变体 | Structure Encoder | Semantic Encoder | Weak Label |
|---|---|---|---|
| `DDE only` | yes | no | shortest hard |
| `Semantic only` | no | yes | shortest hard |
| `DDE + Semantic` | yes | yes | shortest hard |
| `QARAG full` | yes | yes | query-aware soft |

### 实施细节

建议在 config 中显式增加开关，而不是改代码注释：

```yaml
retriever:
  use_structure_encoder: true
  use_semantic_encoder: true

weak_supervision:
  strategy: query_path_soft
```

对应实现位置：

- `retrieve/src/model/retriever.py`
  - `use_structure_encoder=false` 时，不拼接 DDE / topic PE feature；
  - `use_semantic_encoder=false` 时，不计算 `relation_query_sim` / `triple_query_sim`；
  - 保持最终 scorer 都是 shallow MLP。
- `retrieve/configs/retriever/{webqsp,cwq}.yaml`
  - 为每个 ablation 保存独立 config 或命令行 override。

每个 ablation 都跑同样的 `train -> inference -> eval`。

## 实验 3：Weak Supervision Ablation

### 目的

证明改进不是只来自模型结构，而是来自更合理的 weak label。

### Ablation 设置

| 变体 | 说明 |
|---|---|
| `shortest_hard` | 原始 SubgraphRAG：所有 shortest path triples 标为 1 |
| `query_path_selected` | 先构造 path pool，再只保留 query-relation similarity top-M paths，path 内 triples 标为 1 |
| `query_path_soft` | top-M paths 内 triples 使用 path confidence 作为 soft label |

第一版可以先实现 `shortest_hard` 和 `query_path_soft`。`query_path_selected` 可作为中间 ablation，帮助说明 soft label 是否必要。

### 实施细节

当前 QARAG 已有：

```yaml
weak_supervision:
  strategy: query_path_soft
  near_shortest_delta: 0
  top_m_paths: 10
  max_paths_per_pair: 50
  negative_confidence: 0.05
  score_temperature: 1.0
```

建议实验组：

```text
shortest_hard:
  strategy=shortest_hard

query_path_soft:
  strategy=query_path_soft
  near_shortest_delta=0
  top_m_paths=10

near_shortest_soft:
  strategy=query_path_soft
  near_shortest_delta=1
  top_m_paths=10
```

注意：

- `near_shortest_delta=1` 可能导致 CWQ path 数量膨胀，需要记录平均 candidate path 数；
- `max_paths_per_pair` 固定为 50，避免不同 sample 的 path pool 过大；
- 对于多个 query entity / answer entity，path pool 来自所有 pair 的 union。

## 实验 4：K Sensitivity Curve

### 目的

证明 QARAG 在更小 evidence budget 下也能找回有效 triples。

### 设置

使用同一个 `retrieval_result.pth`，在不同 K 上评估：

```text
K = 50, 100, 200, 400, 500
```

画两类曲线：

```text
x-axis: K
y-axis: ans_recall@K

x-axis: K
y-axis: gpt_triple_recall@K
```

### 实施细节

需要确保 `inference.py --max_K >= 500`，否则后续无法评估 `@500`。

建议图中方法：

- `SubgraphRAG`
- `QARAG full`
- `Semantic only`，可选

如果 QARAG 在 `@50/@100` 提升明显，可以作为主论文核心图。

## 实验 5：Breakdown Analysis

### 目的

分析 QARAG 对哪些问题更有效，尤其是 multi-hop 和 multi-topic 问题。

### Breakdown 维度

建议至少做：

```text
1-hop / 2-hop / >=3-hop
single-topic / multi-topic
single-answer / multi-answer
answer-in-graph / answer-not-in-graph
```

### 1-hop / 2-hop / >=3-hop 怎么得到

不要直接用 natural language question 判断 hop 数。应从 processed KG subgraph 计算 topic entity 到 answer entity 的 shortest path distance。

对每个 sample：

1. 读取 `processed/{split}.pkl` 中的：
   - `h_id_list`
   - `r_id_list`
   - `t_id_list`
   - `q_entity_id_list`
   - `a_entity_id_list`
2. 构造有向图 `G`：

```python
G.add_edge(h_id, t_id, triple_id=i, relation_id=r_id)
```

3. 对所有 `(q_entity, answer_entity)` pair 计算两个方向的 shortest path：

```python
q -> answer
answer -> q
```

4. 对每条 path，hop 数定义为：

```text
num_edges = len(path) - 1
```

5. 对一个 sample 的 primary hop label，建议使用：

```text
hop_min = min connected pair shortest distance
```

然后分桶：

```text
hop_min == 1  -> 1-hop
hop_min == 2  -> 2-hop
hop_min >= 3  -> >=3-hop
no path       -> no-path / excluded from hop breakdown
```

为什么用 `hop_min`：

- 多个 topic / answer entity 时，`hop_min` 表示最短可用 reasoning route；
- 和 SubgraphRAG 的 shortest-path weak supervision 更一致；
- 不会因为某个远距离 answer entity 把整个问题误判成困难样本。

同时建议额外记录：

```text
hop_max = max connected pair shortest distance
num_connected_pairs
num_total_pairs
```

如果 reviewer 质疑 `hop_min` 太乐观，可以在 appendix 用 `hop_max` 做补充。

### single-topic / multi-topic 怎么得到

直接根据 processed sample：

```text
num_topic_entities = len(q_entity_id_list)
```

分桶：

```text
num_topic_entities == 1 -> single-topic
num_topic_entities > 1  -> multi-topic
```

注意：

- `q_entity` 可能包含文本名重复，建议使用 `q_entity_id_list` 去重后计数；
- 如果 topic entity 不在 graph 中，应记录为 `topic-missing`，不要混入 normal bucket。

### single-answer / multi-answer 怎么得到

根据：

```text
num_answer_entities = len(set(a_entity_id_list))
```

分桶：

```text
num_answer_entities == 1 -> single-answer
num_answer_entities > 1  -> multi-answer
```

注意：

- 如果 `a_entity_id_list` 为空，说明 answer 不在当前 graph 中，归入 `answer-not-in-graph`；
- `a_entity` 原始文本可能有多个 alias，不建议直接用文本计数。

### answer-in-graph / answer-not-in-graph 怎么得到

使用 inference 输出或 processed sample：

```text
len(a_entity_id_list) > 0
```

或在 `retrieval_result.pth` 中：

```text
len(a_entity_in_graph) > 0
```

分桶：

```text
answer-in-graph
answer-not-in-graph
```

KGQA end-to-end 指标必须分别报告这两个 subset，因为 answer-not-in-graph 时 retriever 不可能覆盖答案实体。

### Breakdown 指标

每个 bucket 内报告：

```text
ans_recall@100
gpt_triple_recall@100
shortest_path_triple_recall@100
```

如果跑 reasoner，则额外报告：

```text
Hit@1
Macro-F1
Hal Score
```

## 实验 6：End-to-End KGQA Sanity

### 目的

证明 retriever 改进能在不改 reasoner 的情况下提升最终 KGQA。

### 设置

固定 SubgraphRAG reasoner：

```text
LLM: meta-llama/Meta-Llama-3.1-8B-Instruct
prompt_mode: scored_100
temperature: 0
frequency_penalty: 0.16
```

只替换 retrieval result：

```bash
cd Experiments/QARAG/reason
python main.py -d webqsp --prompt_mode scored_100 -p ../retrieve/<run_dir>/retrieval_result.pth
python main.py -d cwq   --prompt_mode scored_100 -p ../retrieve/<run_dir>/retrieval_result.pth
```

### 指标

使用 corrected evaluation：

```text
Hit@1
Macro-F1
Macro Precision
Macro Recall
Exact Match
Micro-F1
Hal Score
no-answer ratio
```

### 注意

- 这组实验是 sanity check，不要把论文贡献写成 reasoner 改进；
- LLM 结果受模型版本和 vLLM 环境影响，需要记录 exact model path / revision；
- 如果资源不够，先只跑 `WebQSP + scored_100`。

## 实验 7：Efficiency

### 目的

证明 QARAG 仍然是 lightweight retriever，而不是靠复杂模型换性能。

### 指标

报告：

```text
trainable parameters
training time per epoch
inference time per query
peak GPU memory
average selected top-K serialization size
```

### 实施细节

建议在 `train.py` 和 `inference.py` 中记录：

```python
torch.cuda.max_memory_allocated()
time.perf_counter()
```

对比：

- `SubgraphRAG`
- `QARAG full`

如果 QARAG 只增加 semantic matching scalar feature，参数和时间开销应该很小。

## 推荐论文主表与图

### Main Table

`WebQSP / CWQ` 上的 retrieval 主结果：

```text
Method | ans_recall@100 | shortest_recall@100 | gpt_recall@100 | time/query
```

### Ablation Table

```text
Method | Structure | Semantic | Query-aware label | gpt_recall@100 | ans_recall@100
```

### Figure

K sensitivity：

```text
K = 50, 100, 200, 400, 500
SubgraphRAG vs QARAG
```

### Breakdown Table

```text
Bucket | #Samples | SubgraphRAG gpt_recall@100 | QARAG gpt_recall@100 | Delta
```

Buckets：

- `1-hop`
- `2-hop`
- `>=3-hop`
- `single-topic`
- `multi-topic`
- `answer-in-graph`
- `answer-not-in-graph`

## 最小可执行实验顺序

第一阶段只做 retriever：

1. 跑通 `SubgraphRAG baseline` 和 `QARAG full` 的 `train -> inference -> eval`。
2. 生成 `@50/@100/@200/@400/@500` retrieval table。
3. 跑 `DDE only / Semantic only / DDE + Semantic / QARAG full` ablation。
4. 跑 `shortest_hard / query_path_soft / near_shortest_soft` weak label ablation。
5. 生成 breakdown script，并报告 `1-hop/2-hop/>=3-hop`。

第二阶段再做 end-to-end：

1. 固定 SubgraphRAG reasoner。
2. 只替换 `retrieval_result.pth`。
3. 先跑 `WebQSP scored_100`，确认 trend 后再跑 `CWQ`。

## 风险与注意事项

- `gpt_triple_recall` 依赖 GPT-labeled artifacts，缺失时不能报告该指标。
- `near_shortest_delta=1` 在 CWQ 上可能产生 path explosion，必须记录平均 path pool size。
- hop breakdown 是 graph-distance proxy，不等同于人工标注 reasoning hop，应在论文中说明。
- 多 answer 问题用 `hop_min` 分桶可能偏乐观，appendix 可补 `hop_max`。
- reasoner 结果可能受 LLM backend 和 decoding 参数影响，必须固定并记录。
- 如果 QARAG 提升主要出现在 retrieval 但 KGQA 不提升，需要分析是不是 top-K 中 evidence 顺序、prompt budget 或 LLM reasoning bottleneck 导致。
