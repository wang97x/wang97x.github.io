---
layout: post
title: "MRR@k 与 NDCG@k 详解"
author: "wang"
date: 2025-03-20 15:08:00 +0800
categories: [Information Retrieval]
tags: [SentenceTransformers, Reranking]
math: true
---

在信息检索和推荐系统中，**MRR@k**（Mean Reciprocal Rank at k）和**NDCG@k**（Normalized Discounted Cumulative Gain at k）是两个核心评估指标，用于衡量排序结果的质量。以下是它们的详细对比与应用说明：

---

### **1. MRR@k（平均倒数排名@k）**
#### **定义**
- **目标**：衡量系统返回的排序列表中**第一个相关结果的位置**。
- **公式**：  
  $$
  \text{MRR@k} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}
  $$
  - $$ N $$：查询总数。
  - $$ \text{rank}_i $$：第 $$ i $$ 个查询中第一个相关结果的位置（若前 $$ k $$ 个结果中无相关项，则忽略该查询或计为0）。

#### **特点**
- **优点**：简单直观，强调第一个相关结果的重要性。
- **缺点**：忽略后续相关结果的排序，且对多相关场景不敏感。
- **适用场景**：问答系统、搜索引擎等，用户期望首个结果即相关。

#### **示例**
- 假设有3个查询，其第一个相关结果的位置分别为2、1、5（k=5）：
  $$
  \text{MRR@5} = \frac{1}{3} \left( \frac{1}{2} + \frac{1}{1} + \frac{1}{5} \right) \approx 0.57
  $$

---

### **2. NDCG@k（归一化折损累计增益@k）**
#### **定义**
- **目标**：衡量排序列表中**多级相关性**的累积增益，并归一化到理想排序。
- **公式**：  
  $$
  \text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}
  $$
  - **DCG@k（折损累计增益）**：
    $$
    \text{DCG@k} = \sum_{i=1}^{k} \frac{\text{rel}_i}{\log_2(i+1)}
    $$
    - $$ \text{rel}_i $$：第 $$ i $$ 个位置的结果的相关性分数（如0/1或分级标签）。
  - **IDCG@k（理想DCG）**：将相关性分数降序排列后的DCG@k。

#### **特点**
- **优点**：支持多级相关性，考虑排序位置对增益的折损。
- **缺点**：需定义相关性分数，计算复杂度较高。
- **适用场景**：推荐系统、文档排序等，需关注整体排序质量。

#### **示例**
- 假设排序结果为 `[3, 2, 3, 0, 1]`（相关性分数，k=5）：
  - **DCG@5**：
    $$
    \frac{3}{\log_2 2} + \frac{2}{\log_2 3} + \frac{3}{\log_2 4} + 0 + \frac{1}{\log_2 6} \approx 3 + 1.26 + 1.5 + 0 + 0.39 = 6.15
    $$
  - **IDCG@5**（理想排序 `[3, 3, 2, 1, 0]`）：
    $$
    \frac{3}{\log_2 2} + \frac{3}{\log_2 3} + \frac{2}{\log_2 4} + \frac{1}{\log_2 5} + 0 \approx 3 + 1.89 + 1 + 0.43 = 6.32
    $$
  - **NDCG@5**：
    $$
    \frac{6.15}{6.32} \approx 0.97
    $$

---

### **3. 对比总结**

| **特性**      | **MRR@k**  | **NDCG@k**   |
|-------------|------------|--------------|
| **关注点**     | 第一个相关结果的位置 | 整体排序质量与多级相关性 |
| **相关性支持**   | 二元（相关/不相关） | 多级（如0-5分）    |
| **计算复杂度**   | 低          | 高（需计算理想排序）   |
| **典型应用**    | 问答系统、搜索引擎  | 推荐系统、个性化排序   |
| **对位置的敏感度** | 仅关注首个相关结果  | 对靠前位置赋予更高权重  |

---

### **4. 如何选择？**
- **选择 MRR@k**：  
  - 用户需求是快速找到**至少一个**正确答案（如搜索引擎）。  
  - 任务中相关性判定为二元（相关/不相关）。
- **选择 NDCG@k**：  
  - 需要精细化评估**多相关结果**的排序质量（如推荐系统）。  
  - 支持多级相关性标签（如“完美相关”“部分相关”）。

---

### **5. 实际应用技巧**
1. **设定合理的k值**：  
   - 搜索引擎通常关注前10（k=10），推荐系统可能关注前20（k=20）。
2. **处理无相关结果**：  
   - MRR@k中可忽略无相关结果的查询，或统一赋予0分。
3. **相关性分数定义**：  
   - NDCG@k中需明确分级标准（如点击率、人工标注）。

---

### **6. 代码实现示例（Python）**
```python
import numpy as np

# MRR@k计算
def calculate_mrr(rankings, k):
    reciprocal_ranks = []
    for ranks in rankings:
        for i in range(min(k, len(ranks))):
            if ranks[i] == 1:  # 假设1表示相关
                reciprocal_ranks.append(1/(i+1))
                break
        else:
            reciprocal_ranks.append(0)
    return np.mean(reciprocal_ranks)

# NDCG@k计算
def calculate_ndcg(scores, ideal_scores, k):
    def dcg(scores):
        return sum( (score / np.log2(i+2)) for i, score in enumerate(scores[:k]) )
    dcg_val = dcg(scores)
    idcg_val = dcg(sorted(ideal_scores, reverse=True))
    return dcg_val / idcg_val if idcg_val > 0 else 0
```

---

### **总结**
- **MRR@k**：简单高效，适合快速验证首个相关结果的质量。  
- **NDCG@k**：全面精准，适合多相关性和精细化排序场景。  
- **联合使用**：实际系统中可先用MRR@k筛选模型，再用NDCG@k优化排序细节。
