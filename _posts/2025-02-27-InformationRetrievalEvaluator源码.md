---
layout: post
title: "InformationRetrievalEvaluator源码"
author: "wang"
date: 2025-02-27 09:49:00 +0800
categories: [SentenceTransformers]
tags: [InformationRetrievalEvaluator ]
---
# InformationRetrievalEvaluator类解析与总结

## 简介

`InformationRetrievalEvaluator`类是UKPLab/sentence-transformers库中的一部分，用于在信息检索（IR）设置中评估模型的性能。该类通过给定的一组查询，从大型语料库中检索每个查询最相似的前k个文档，并测量多种IR指标如平均互惠排名（MRR）、召回率@k和归一化折扣累积增益（NDCG）。

## 类结构

### 属性

- `queries`：查询字典，映射查询ID到查询文本。
- `corpus`：语料库字典，映射文档ID到文档文本。
- `relevant_docs`：字典，映射查询ID到相关文档ID的集合。
- `corpus_chunk_size`：每个语料库块的大小。
- `mrr_at_k`：用于MRR计算的k值列表。
- `ndcg_at_k`：用于NDCG计算的k值列表。
- `accuracy_at_k`：用于精度计算的k值列表。
- `precision_recall_at_k`：用于精度和召回计算的k值列表。
- `map_at_k`：用于MAP计算的k值列表。
- `show_progress_bar`：是否显示进度条。
- `batch_size`：评估的批量大小。
- `name`：评估名称。
- `write_csv`：是否将评估结果写入CSV文件。
- `truncate_dim`：截断嵌入的维度。
- `score_functions`：字典，映射分数函数名称到分数函数。
- `main_score_function`：主要分数函数。
- `query_prompt`：编码查询时使用的提示。
- `query_prompt_name`：提示的名称。
- `corpus_prompt`：编码语料库时使用的提示。
- `corpus_prompt_name`：提示的名称。

### 方法

#### `__init__`

初始化`InformationRetrievalEvaluator`类的实例。设置查询、语料库、相关文档及各种评估参数。

```python
def __init__(
    self,
    queries: dict[str, str],  # qid => query
    corpus: dict[str, str],  # cid => doc
    relevant_docs: dict[str, set[str]],  # qid => Set[cid]
    corpus_chunk_size: int = 50000,
    mrr_at_k: list[int] = [10],
    ndcg_at_k: list[int] = [10],
    accuracy_at_k: list[int] = [1, 3, 5, 10],
    precision_recall_at_k: list[int] = [1, 3, 5, 10],
    map_at_k: list[int] = [100],
    show_progress_bar: bool = False,
    batch_size: int = 32,
    name: str = "",
    write_csv: bool = True,
    truncate_dim: int | None = None,
    score_functions: dict[str, Callable[[Tensor, Tensor], Tensor]] | None = None,
    main_score_function: str | SimilarityFunction | None = None,
    query_prompt: str | None = None,
    query_prompt_name: str | None = None,
    corpus_prompt: str | None = None,
    corpus_prompt_name: str | None = None,
) -> None:
    ...
```

#### `__call__`

执行评估并计算评估指标。

```python
def __call__(self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs) -> dict[str, float]:
    ...
```

#### `compute_metrics`

从结果中计算各种IR指标。

```python
def compute_metrics(self, queries_result_list: list[object]):
    ...
```

#### `output_scores`

记录计算的分数。

```python
def output_scores(self, scores):
    ...
```

## 结论

`InformationRetrievalEvaluator`类通过提供一系列方法和属性，方便用户在信息检索设置中评估模型的性能。它可以处理查询和大型语料库，并计算多种IR指标，为模型性能的定量评估提供了强有力的工具。


以下是对 `compute_metrices` 方法的详细解析：

### **方法目标**
`compute_metrices` 方法的主要目标是计算信息检索（Information Retrieval, IR）任务的性能指标。它通过比较查询（queries）与语料库（corpus）中的文档嵌入向量，评估模型在检索相关文档时的性能。

### **参数说明**
- **`model`**: 嵌入模型，用于编码查询和文档。
- **`corpus_model`**: 可选的语料库模型，如果未提供，则默认与 `model` 使用相同模型。
- **`corpus_embeddings`**: 预先计算的语料库嵌入向量。如果未提供，则将在方法中重新计算。

### **代码结构**
1. **定义变量和初始化**
2. **计算查询嵌入向量**
3. **分块处理语料库**
4. **计算相似度并选择前 N 个结果**
5. **整理结果并计算指标**
6. **输出指标**

### **详细解析**

#### **1. 定义变量和初始化**
```python
max_k = max(
    max(self.mrr_at_k),
    max(self.ndcg_at_k),
    max(self.accuracy_at_k),
    max(self.precision_recall_at_k),
    max(self.map_at_k),
)
```
- 获取当前使用的最大 `k` 值。`k` 是在计算 `MRR@k`, `NDCG@k`, `Accuracy@k`, `Recall@k` 等指标时所用的最大值。

#### **2. 计算查询嵌入向量**
```python
with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(self.truncate_dim):
    query_embeddings = model.encode(
        self.queries,
        prompt_name=self.query_prompt_name,
        prompt=self.query_prompt,
        batch_size=self.batch_size,
        show_progress_bar=self.show_progress_bar,
        convert_to_tensor=True,
    )
```
- 使用 `model` 编码查询文本，生成查询的嵌入向量。
- 如果指定了 `self.truncate_dim`，则嵌入向量会被截断（减少维度）。

#### **3. 分块处理语料库**
```python
for corpus_start_idx in trange(
    0, len(self.corpus), self.corpus_chunk_size, desc="Corpus Chunks", disable=not self.show_progress_bar
):
    corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))
```
- 为避免内存不足，将语料库分成多个大小为 `self.corpus_chunk_size` 的块进行处理。

#### **4. 编码语料库块**
```python
if corpus_embeddings is None:
    with (
        nullcontext()
        if self.truncate_dim is None
        else corpus_model.truncate_sentence_embeddings(self.truncate_dim)
    ):
        sub_corpus_embeddings = corpus_model.encode(
            self.corpus[corpus_start_idx:corpus_end_idx],
            prompt_name=self.corpus_prompt_name,
            prompt=self.corpus_prompt,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
        )
else:
    sub_corpus_embeddings = corpus_embeddings[corpus_start_idx:corpus_end_idx]
```
- 如果未提供 `corpus_embeddings`，则使用 `corpus_model` 编码当前语料库块。
- 否则，从预存的 `corpus_embeddings` 中提取当前块。

#### **5. 计算相似度并选择前 N 个结果**
```python
pair_scores = score_function(query_embeddings, sub_corpus_embeddings)
pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(
    pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=False
)
```
- 使用指定的评分函数（如余弦相似度）计算查询和当前块语料库文档的相似度。
- 获取每个查询前 `max_k` 个最高分文档的索引和分数。

#### **6. 维护前 N 个结果**
```python
for query_itr in range(len(query_embeddings)):
    for sub_corpus_id, score in zip(pair_scores_top_k_idx[query_itr], pair_scores_top_k_values[query_itr]):
        corpus_id = self.corpus_ids[corpus_start_idx + sub_corpus_id]
        if len(queries_result_list[name][query_itr]) < max_k:
            heapq.heappush(queries_result_list[name][query_itr], (score, corpus_id))
        else:
            heapq.heappushpop(queries_result_list[name][query_itr], (score, corpus_id))
```
- 使用 `heapq` 包维护每个查询的前 `max_k` 个结果，确保只保留分数最高的文档。

#### **7. 整理结果**
```python
for name in queries_result_list:
    for query_itr in range(len(queries_result_list[name])):
        for doc_itr in range(len(queries_result_list[name][query_itr])):
            score, corpus_id = queries_result_list[name][query_itr][doc_itr]
            queries_result_list[name][query_itr][doc_itr] = {"corpus_id": corpus_id, "score": score}
```
- 将结果整理成可读的字典格式，包含文档 ID 和相似度分数。

#### **8. 打印日志**
```python
logger.info(f"Queries: {len(self.queries)}")
logger.info(f"Corpus: {len(self.corpus)}\n")
```
- 打印查询和语料库的数量信息。

#### **9. 计算指标**
```python
scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}
```
- 调用 `self.compute_metrics` 方法计算每个评分函数对应的指标（如 MRR、NDCG 等）。

#### **10. 输出指标**
```python
for name in self.score_function_names:
    logger.info(f"Score-Function: {name}")
    self.output_scores(scores[name])
```
- 打印每个评分函数对应的评价指标。

### **输出结果**
```python
return scores
```
- 返回一个字典，包含不同评分函数对应的评价指标。

### **总结**
`compute_metrics` 方法通过以下步骤计算信息检索任务的性能指标：
1. 计算查询和语料库的嵌入向量。
2. 分块处理语料库，以计算相似度。
3. 维护每个查询的前 N 个结果。
4. 定期评估指标（如 MRR、NDCG 等）。

通过这种方式，该方法可以高效地评估嵌入模型在特定信息检索任务中的性能。