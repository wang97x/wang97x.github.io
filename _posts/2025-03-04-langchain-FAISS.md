---
layout: post
title: "Langchain-FAISS"
author: "wang"
date: 2025-03-04 09:00:00 +0800
categories: [Langchain]
tags: [faiss]
---


```python
class FAISS(VectorStore):
    def __init__(
        self,
        embedding_function: Union[
            Callable[[str], List[float]],
            Embeddings,
        ],
        index: Any,
        docstore: Docstore,
        index_to_docstore_id: Dict[int, str],
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        normalize_L2: bool = False,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
    ):
...
```
## 参数解读
### relevance_score_fn：

`relevance_score_fn` 是一个参数，用于指定一个函数，该函数将相似度分数转换为相关性分数。这个参数通常用于信息检索和文本相似度计算的场景中，特别是在使用向量存储（如FAISS）进行相似度搜索时。以下是 `relevance_score_fn` 参数的详细含义：

### 1. 定义和作用
`relevance_score_fn` 是一个函数，它将原始的`相似度分数（通常是距离或相似度度量）`转换为一个标准化的相关性分数，范围在 [0, 1] 之间。0 表示不相关，1 表示最相关。

### 2. 使用场景
在信息检索中，`relevance_score_fn` 用于将向量存储返回的相似度分数转换为更直观的相关性分数。这有助于用户更好地理解检索结果的相关性。

### 3. 计算方式
`relevance_score_fn` 的具体实现取决于所使用的距离或相似度度量。常见的计算方式包括：
- **余弦相似度**：`relevance_score = 1.0 - distance`。
- **欧式距离**：`relevance_score = 1.0 - distance / sqrt(2)`。
- **内积**：`relevance_score = 1.0 - distance`（如果距离为负值，则取其相反数）。

### 4. 重要性
选择合适的 `relevance_score_fn` 对于确保相关性分数的准确性和可解释性至关重要。不同的距离度量和嵌入向量的缩放方式可能需要不同的相关性分数函数。

### 总结
`relevance_score_fn` 是一个用于将相似度分数转换为相关性分数的函数，它在信息检索和文本相似度计算中起到关键作用。通过选择合适的 `relevance_score_fn`，可以确保相关性分数的准确性和可解释性。
