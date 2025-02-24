---
layout: post
title: "Transformer源码-位置编码(PositionalEncoder)"
author: "wang"
date: 2025-02-22 13:38:00 +0800
categories: [LLM]
tags: [Transformer]
---

Transformer模型自2017年提出以来，已经成为自然语言处理（NLP）领域的主流模型。与传统的循环神经网络（RNN）不同，Transformer模型完全基于自注意力机制，因此在处理长距离依赖关系方面有显著优势。然而，由于Transformer模型缺乏内置的序列顺序信息，必须通过位置编码（Positional Encoding）显式引入位置信息，以便模型能够区分序列中的不同位置。
## 位置编码
位置编码是Transformer模型中一个至关重要的部分，直接影响到模型对序列信息的处理能力。

## 常见位置编码
本文将系统地介绍Transformer模型中的三种主要位置编码方法：绝对位置编码、相对位置编码和旋转位置编码。

#### 绝对位置编码
1. 实现方式
绝对位置编码（Absolute Positional Encoding）是最常见的一种位置编码方法，其思想是在每个输入序列的元素上添加一个位置向量，以表示该元素在序列中的具体位置。这个位置向量通常通过固定的函数生成，与输入数据无关。通常使用的是正弦和余弦函数，这样生成的编码具有很强的周期性，能够捕捉序列中的相对位置信息。
对于序列中的第 \(pos\) 个位置，绝对位置编码向量的第 \(i\) 个维度的值定义如下：

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
$$

$$
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
$$

其中，$pos$是词在序列中的位置，$d_model$是嵌入的维度。 

2. 代码实现

```python
import numpy as np

def get_absolute_positional_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe
```

3. 案例分析
在上面的代码中，我们为长度为8的句子生成了一个绝对位置编码矩阵。该矩阵的维度为（8, 32），每一行表示句子中一个位置的编码。通过热图可以看到，不同位置的编码在特定维度上具有不同的模式，这些模式帮助Transformer区分序列中不同位置的元素。
4. 优缺点
绝对位置编码的优势在于其简单且具有良好的可解释性。它能够有效地为序列中的每个位置分配独特的编码，从而帮助模型捕捉序列的顺序信息。然而，它也有一定的局限性，尤其是在处理变长序列或长距离依赖时，绝对位置编码可能无法充分表达复杂的位置信息。

### 相对位置编码
1. 实现方式
相对位置编码（Relative Positional Encoding）并不直接为每个位置分配一个唯一的编码，而是关注序列中各元素之间的相对位置。相对位置编码的核心思想是通过计算序列中元素之间的距离，来表示它们之间的相对关系。这种方法尤其适合处理需要捕捉长距离依赖关系的任务，因为它能够更加灵活地表示序列中的结构信息。
相对位置编码可以通过多种方式实现，其中最常用的方法之一是将位置差值与注意力权重相结合，即在计算自注意力时，不仅考虑内容，还考虑位置差异。 
2. 代码实现

```python
import torch
import torch.nn.functional as F

class RelativePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        # 生成相对位置编码
        self.relative_positions_matrix = self.generate_relative_positions_matrix(max_len)
        self.embeddings_table = self.create_embeddings_table(max_len, d_model)
    
    def generate_relative_positions_matrix(self, length):
        range_vec = torch.arange(length)
        distance_mat = range_vec[None, :] - range_vec[:, None]
        return distance_mat
    
    def create_embeddings_table(self, max_len, d_model):
        table = torch.zeros(max_len, max_len, d_model)
        for pos in range(-max_len+1, max_len):
            table[:, pos] = self.get_relative_positional_encoding(pos, d_model)
        return table
    
    def get_relative_positional_encoding(self, pos, d_model):
        pos_encoding = torch.zeros(d_model)
        for i in range(0, d_model, 2):
            pos_encoding[i] = torch.sin(pos / (10000 ** ((2 * i)/d_model)))
            if i + 1 < d_model:
                pos_encoding[i + 1] = torch.cos(pos / (10000 ** ((2 * i)/d_model)))
        return pos_encoding
    
    def forward(self, length):
        positions_matrix = self.relative_positions_matrix[:length, :length]
        return F.embedding(positions_matrix, self.embeddings_table)
```

3. 案例分析
在这个示例中，我们生成了一个基于相对位置的编码矩阵，该矩阵的维度为（8, 8, 32），每个元素表示句子中两个位置之间的相对编码向量。这种编码方式在处理长句子时能够更好地捕捉不同元素之间的关系，因为它可以灵活地处理序列中的相对位置。
4. 优缺点
相对位置编码的优势在于其对序列长度和相对位置信息的良好适应性，特别适合处理长文本或存在复杂依赖关系的任务。然而，相对位置编码的实现相对复杂，且在某些情况下可能增加计算成本。

### 旋转位置编码
1. 实现方式
旋转位置编码（Rotary Positional Encoding, RoPE）是近年来提出的一种新型位置编码方法。RoPE的核心思想是通过对输入向量进行旋转变换，将位置信息嵌入到向量中。具体来说，RoPE通过旋转每个维度对中的向量，实现对序列中位置信息的编码。
RoPE具有很强的表达能力，尤其是在处理具有对称性或周期性的任务时，能够更加自然地捕捉序列中的位置信息。
2. 代码实现

```python
import torch

def rotate_every_two(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    x = torch.cat([-x2, x1], dim=-1)
    return x

def apply_rotary_pos_emb(x, sin, cos):
    return x * cos + rotate_every_two(x) * sin
```

3. 案例分析
在上述代码中，我们通过旋转向量对实现了RoPE编码，并将其应用于Q和K矩阵。可视化结果显示，经过旋转位置编码后，Q矩阵的不同维度展示出明显的周期性模式，这有助于捕捉序列中的周期性或对称性信息。
4. 优缺点
RoPE的优势在于其强大的表达能力，特别是在处理具有对称性或周期性特征的数据时表现优异。此外，RoPE具有一定的灵活性，可以应用于不同类型的输入数据。然而，RoPE的实现复杂度较高，且其适用性尚需在更多场景中验证。

## 总结
位置编码是Transformer模型中至关重要的一部分，不同的编码方式适用于不同的任务和数据类型。本文详细介绍了绝对位置编码、相对位置编码和旋转位置编码的原理、实现及应用，通过具体的案例分析展示了它们在实际任务中的表现。随着NLP领域的不断发展，新的位置编码方法可能会不断涌现，进一步提升Transformer模型在复杂任务中的表现。
