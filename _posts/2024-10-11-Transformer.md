---
layout: post 
title: "Transformer系列"
author: "wang"
date: 2024-10-11 11:27:00 +0800
categories: [Transformer]
tags: [python, transformer, llm]
---
# Transformer系列-基础架构
## Transformer架构：
Transformer是一种用于自然语言处理（NLP）和其他序列到序列（sequence-to-sequence）任务的深度学习模型架构，它在2017年由Vaswani等人首次提出。Transformer架构引入了自注意力机制（self-attention mechanism），这是一个关键的创新，使其在处理序列数据时表现出色。
以下是Transformer的一些重要组成部分和特点：
    - 自注意力机制（Self-Attention）：这是Transformer的核心概念之一，它使模型能够同时考虑输入序列中的所有位置，而不是像循环神经网络（RNN）或卷积神经网络（CNN）一样逐步处理。自注意力机制允许模型根据输入序列中的不同部分来赋予不同的注意权重，从而更好地捕捉语义关系。
- 多头注意力（Multi-Head Attention）：Transformer中的自注意力机制被扩展为多个注意力头，每个头可以学习不同的注意权重，以更好地捕捉不同类型的关系。多头注意力允许模型并行处理不同的信息子空间。
- 堆叠层（Stacked Layers）：Transformer通常由多个相同的编码器和解码器层堆叠而成。这些堆叠的层有助于模型学习复杂的特征表示和语义。
- 位置编码（Positional Encoding）：由于Transformer没有内置的序列位置信息，它需要额外的位置编码来表达输入序列中单词的位置顺序。
- 残差连接和层归一化（Residual Connections and Layer Normalization）：这些技术有助于减轻训练过程中的梯度消失和爆炸问题，使模型更容易训练。
- 编码器和解码器：Transformer通常包括一个编码器用于处理输入序列和一个解码器用于生成输出序列，这使其适用于序列到序列的任务，如机器翻译。
## Transformer的结构：
Transformer的编码组件是由6个编码器叠加在一起组成的，解码器同样如此。所有的编码器在结构上是相同的，但是它们之间并没有共享参数。如下图所示：
[图片]
结构拆解：
[图片]
1. 输入嵌入和位置编码（红色框）
- 输入嵌入：这是将输入单词（或标记）转换为固定大小的密集向量的第一步。这些向量以模型可以处理的方式表示输入单词。
- 位置编码：由于 Transformer 模型没有任何固有的顺序感知（与 RNN 不同），因此位置编码被添加到输入嵌入中。它注入有关句子中每个单词位置的信息，使模型能够理解单词的顺序。
常用的位置编码
- 输出嵌入(右移)：解码器的输出（outputs）在训练阶段通常会进行右移（shifted right）操作。这意味着在训练时，解码器的输入是目标序列的前一个词，而不是当前词。这样做的目的是为了让解码器在预测下一个词时，只能使用它之前已经生成的词，而不能使用未来的词，从而模拟自回归的语言模型。
[图片]
在这里，引入了两个新的token，分别是<|im_start|>和<|im_end|>，此外，解码器一次只能接受一个token作为输入，也就是说，<|im_start|>会被作为一个输入，而"太”就是下一个预测token。
2. 编码器块（蓝色框）
- 编码器块由多个相同的层组成（表示为$$N \times$$）。每个编码器层都有两个主要组件：
   多头注意力：该层允许模型通过并行计算多个注意力分数（每个头关注输入的不同部分）来同时关注输入序列的不同部分。
   [图片]

   前馈网络：独立应用于每个位置的全连接神经网络，转换注意力层的输出。
   [图片]

Transformer前馈神经网络两层结构： 包括两个线性变换，并在它们之间使用ReLU激活函数。 两个线性层的差异主要体现在它们的作用和维度变化上。
第一层线性变换负责将输入映射到更高维度的空间，并引入非线性；而第二层线性变换则负责将输出映射回与输入相同的维度（或兼容的维度），通常不引入额外的非线性。
1. 第一层线性变换：这是一个全连接层，它接收自注意力层的输出作为输入，并将其映射到一个更高维度的空间。这个步骤有助于模型学习更复杂的特征表示。
2. 激活函数：在第一层全连接层之后，通常会应用一个非线性激活函数，如ReLU（Rectified Linear Unit）。ReLU函数帮助模型捕获非线性关系，提高模型的表达能力。
3. 第二层线性变换：这也是一个全连接层，它将前一层的输出映射回与输入相同的维度（或与模型其他部分兼容的维度）。这一层通常没有非线性激活函数。
- Add&Norm：注意力子层和前馈子层后面都跟着残差连接（将层的输入添加到其输出）和层规范化以稳定训练。

3. 解码器块（绿色框）
[图片]
- 解码器还由多个相同的层（$$N \times$$）组成。每个解码器层包含：
  屏蔽多头注意力：与编码器的多头注意力类似，但屏蔽确保模型无法关注未来的位置，从而强制语言生成中的因果关系。
  [图片]
  多头注意力：这允许解码器关注编码器的输出，重点关注输入序列的相关部分。
  前馈网络：与编码器一样，它在每个位置应用变换。
- 解码器还在每个子层之后使用 Add & Norm。

4. 输出层（粉色框）
- 经过解码器块后，输出经过线性层，将输出向量映射到词汇的大小。
- 然后，softmax 层将这些向量转换为词汇表上的概率分布，表示模型对下一个单词预测的置信度。
Encoder-Decoder\Encoder-only\Decoder-only结构：
Encoder-only
Encoder-only是以Bert为代表的模型及其衍生优化版本为主，那就以Bert为例来学习Encoder-only架构；
BERT（Bidirectional Encoder Representations from Transformers）是一种在自然语言处理（NLP）领域引起巨大轰动的预训练语言模型，由Google于2018年提出。其核心原理是结合了Transformer架构和双向语言模型预训练策略，使得模型能够更好地理解和表征文本数据。BERT的出现在NLP领域引起了巨大的轰动，因为它在各种NLP任务上取得了巨大成功，同时也开创了一种新的预训练模型的范式。源码：https://github.com/google-research/bert。
BERT原理：
1. Transformer 架构：BERT基于Transformer模型。BERT仅使用编码器部分，因为它主要用于处理单向输入数据。Transformer的核心是自注意力机制（Self-Attention），它允许模型在编码输入序列时同时考虑序列中的所有位置，而无需将注意力限制在固定大小的窗口内。自注意力机制使得模型能够在不同位置之间建立关联，从而更好地理解上下文信息。
2. 双向语言模型预训练：传统的语言模型通常是单向的，只能从左到右或者从右到左预测文本。而BERT采用了双向语言模型预训练的策略，能够同时考虑文本序列中的左右上下文信息。这使得模型在预训练阶段能够更好地捕捉文本的语义和语法信息，从而生成更加丰富的语言表示。
    在双向语言模型预训练中，BERT采用了两个任务：
    1. Masked Language Model（MLM）：模型随机地掩盖输入文本中的一部分词语，然后尝试预测这些被掩盖的词语。这使得模型必须基于其上下文来推断被掩盖的词语，从而学习双向的语言表示。
    2. Next Sentence Prediction（NSP）：模型接受两个句子作为输入，并尝试预测这两个句子是否是连续的。这个任务可以帮助模型理解文本之间的逻辑关系和连贯性。
    [图片]
    左侧的预训练部分展示了BERT如何在无标签数据上进行训练。图中显示了两个任务：下一个句子预测（Next Sentence Prediction, NSP）和掩码语言模型（Masked Language Model, Mask LM）。在NSP任务中，模型预测两个文本片段是否在原始文档中相邻。在Mask LM任务中，BERT预测输入中被[MASK]标记替换的原始词语。
    右侧显示了微调过程。在这里，预训练的BERT模型被调整以适应特定任务，本例中涉及对一个名为SQuAD的数据集进行问答。在BERT之上添加了一个额外的层，通常是一个分类器，用于特定任务。在微调期间，所有参数都一起微调。
    两个图中都显示了输入中的标记：[CLS]，用于分类任务的特殊标记，以及[SEP]，用于分隔段落的标记。此外，在右侧，带有注释的箭头表示模型预测文本中答案范围的“开始”和“结束”。
3. 多层嵌套结构：BERT使用了多层Transformer编码器，并且在这些层之间引入了不同的任务特定变换。每个编码器由多头自注意力层和前馈神经网络层组成，以及一些残差连接和层归一化层。这种多层嵌套结构使得BERT能够学习多层次、多粒度的语言表示，适应不同任务的需求。
一句话总结，BERT核心原理：使用多层嵌套的Transformer的编码器来处理输入序列，使用双向语言模型预训练策略进行掩码预测；到这里大家可能想问了，为什么要用这种框架，Bert开始的时候只是希望能够用这个框架能够学习语言的语法规则，针对主要是文本分类、问答等任务，所以只需要使用Transformer的编码器能够实现文本的语义理解就可以了，不需要生成序列。
搞清楚了Bert原理，那为什么说BERT属于Encoder-only模型？
很简单，因为它只使用了Transformer模型中的编码器部分，而没有使用解码器。在Transformer模型中，编码器负责将输入序列转换为上下文感知的表示，而解码器则负责生成输出序列。BERT使用了编码器。只使用编码器最主要的原因：BERT的预训练目标是通过掩盖部分输入来预测其他部分，或者预测两个句子之间的关系，这些任务并不涉及到生成输出序列，因此不需要解码器。上面我们已经说Transformer的解码器的作用，不懂请回看；
那分析一下BERT的存在的问题：
1. 大型模型需求：BERT及其衍生模型通常需要大量的计算资源和存储空间来训练和部署。大型模型需要更多的参数和更多的计算，这会导致训练和推理的成本较高，限制了它们在资源有限的环境中的应用。
2. 上下文理解能力有限：尽管BERT在一定程度上能够理解输入文本的上下文信息，但它并不能完全解决长距离依赖问题。在处理长文本或长距离依赖关系的任务时，BERT可能会出现性能下降或信息丢失的情况。
3. 缺乏对抗性：BERT模型在处理对抗性样本（adversarial examples）时表现较差，容易受到攻击和干扰。对抗性样本是经过微小修改的输入，对人类来说很难察觉，但可以导致模型输出错误。
4. 预训练数据偏差：BERT模型通常是在大规模通用文本数据上进行预训练的，这可能导致模型在某些特定领域或任务上的表现不佳。如果任务领域与预训练数据的领域不匹配，模型可能需要更多的微调或领域适应。
5. 无法处理多个任务的动态适应性：尽管BERT模型可以通过微调来适应特定的下游任务，但在处理多个任务时，模型的动态适应性有限。因此，对于多任务学习或者需要动态适应不同任务的场景，BERT可能并不是最佳选择。
6. 固定长度输入限制：BERT模型的输入长度有限，通常为512个token，这限制了模型处理较长文本的能力。对于长文本或长句子，需要进行截断或分段处理，可能会损失一些上下文信息。
Encoder-Decoder
从上面的图中我们可以看到Encoder-Decoder架构的模型有T5、GLM等，为了能够让更多的人看懂，我们就以清华大学的GLM为例来继续，GLM的全称基于自回归空白填充预训练框架（General Language Model Pretraining with Autoregressive Blank Infilling），这个框架的思路，结合BERT的思路，从输入文本中随机地空白出连续的跨度的token，并按照自回归预训练的思路，训练模型来依次重建这些跨度。GLM掩码的不是一个单词或是一个实体，而是一个句子片段。
[图片]
这里出现了一个新的名词：自回归；自回归的核心思想是使用当前时刻的输入来预测下一个时刻的输出。在语言模型中，输入通常是文本序列的前一部分，而输出是下一个词或字符。大家只需要理解这一点就可以了，下一章节我们会详细说明；
GLM最初是为了能够NLP的任务更加全面的解决。所以借鉴了BERT的思路对输入的表示进行建模，同时采用了解码器构建模型的生成能力，主要的做法，请结合下图阅读：
[图片]
- 对输入X进行随机采样，采样结果进行Mask替换，如图：原始文本x1x2x3x4x5x6中的 x3 和 x5x6 分别用 [M]替换，变成 x1x2[M]x4[M]，这个就是Part A
- 把Part A中所有 [M]位置的token提取出来并对顺序打乱，并且添加前缀 [S]，得到seq B。：[S]x5x6[S]x3
- 把Part A和Part B拼接起来，输入GLM，GLM中Part A采用双向编码器，如上右图(d)的蓝色框内。Part B是单向解码器，可以看到Part A的所有内容，以及当前时间片之前的内容，但它不能看到当前时间片之后的内容，如上图右图(d)的红绿框。
关键点:
1. 在掩码处理中，从输入数据中随机选择一部分数据进行掩码。要服从泊松分布，重复采样，直到原始tokens中有15%被mask。
2. 在GLM模型中，采用了自回归空白填充的自监督训练方式。需要设计掩码策略来生成掩码，如根据预先设定的规则来选择掩码的长度和位置。
3. 在生成掩码后，需要对掩码进行填充。在GLM模型中，采用了特殊的填充方式，如span shuffling和2D positional encoding。
4. 损失函数：在掩码处理过程中，需要根据损失函数来计算掩码处理的效果。在GLM模型中，采用了交叉熵损失函数来衡量模型在掩码处理任务上的表现。这个过程涉及到优化理论和数值分析的知识。
5. 自回归环节有位置编码环节，将位置编码的序列分为两个组，这两个组分别是：
  1. Part A编码：1、mask后的序列按照位置生成位置编码信息；2、采样后的序列在原始序列中的位置编码信息
  2. Part B编码：1、将采样多个子序列随机排列，然后每个子序列各自进行排序，互不干扰；因为输入序列是经过[mask]的原始序列+采样生成的序列，为了保持编码序列一致性，在这里原始序列的位置编码全部为0
Decoder-only:
现在最热门就是这个架构了，解码器结构，当家的应该也是目前整个大模型领域的领头羊：GPT，作为大模型领域当之无愧的领头羊，很多人对他的原理应该都了解一点，在这里我们就挑重点了，毕竟我们的终极目的是为了探讨他们是不是智能的，写到现在我都有点忘记这个终极目的了，快变成一篇技术科普了，当然也是要让大家明白这些不同架构之间的特点和区别；
GPT:
GPT（Generative Pre-trained Transformer）是由OpenAI提出的一种预训练语言模型，它基于Transformer架构的解码器部分。GPT的核心原理包括以下几个关键点：
1. Transformer架构：GPT模型采用了Transformer的解码器结构，这种结构特别适合处理序列数据，并且能够捕捉到序列中的长距离依赖关系。
2. 自回归任务：在预训练阶段，GPT使用了自回归任务，即给定文本序列的前面部分，预测下一个词或字符。这种任务使得模型能够学习到语言的统计特性和序列中的顺序关系。
3. 多层Transformer解码器：GPT模型由多层Transformer解码器堆叠而成，每层解码器都包含多头自注意力机制和前馈神经网络。这种多层结构使得模型能够在不同的抽象层次上捕捉文本的特征。
4. 掩码注意力（Masked Attention）：在GPT中，由于模型需要预测下一个词，因此会使用掩码自注意力机制。这种机制通过掩盖未来的信息，使得模型只能利用当前和过去的信息来生成下一个词，这与自回归语言模型的特点相符合。
5. 预训练和微调：GPT模型首先在大规模的文本数据集上进行预训练，学习通用的语言表示。然后，可以通过微调（Fine-tuning）的方式，将预训练好的模型应用到特定的下游任务中，如文本分类、问答等。
[图片]

[图片]
GPT在最初的目的就是为了生成任务而诞生，所以它采用了解码器框架。详细的过程：https://jalammar.github.io/illustrated-gpt2/
LLAMA:
1. 模型结构
  1. 标准化：为了提高训练的稳定性，标准化每个transformer子层输入来替换原始标准化输出；（Open pre-trained transformer language models）
  2. 使用PaLM中的SwiGLu作为激活函数，使用SwiGLU来代替Relu，dinmension由PaLM的4d->2/3*4d
  3. 旋转EMbedding：采用旋转Embedding来代替绝对位置Embedding
2. 优化器
  1. 使用AdamW作为优化器，优化器参数：$$β_1 = 0.9, β_2 = 0.95$$
  2. 学习率采用cos学习率机制，最终的学习率是最大学习率的10%
  3. 衰减权重：0.1；梯度裁剪:1;warmup:2000steps
3. 提效
  1. 采用具有因果关系的multi-head-attention来降低内存和训练时间
  2. 为了进一步提高训练效率，我们减少了在使用检查点的向后传递过程中重新计算的激活量
注意力机制：

Normalization 算法：
LayerNorm：
层归一化(LayerNorm)对Transformer等模型来说非常重要，它可以帮助稳定训练并提升模型收敛性。LayerNorm针对一个样本所有特征计算均值和方差，然后使用这些来对样本进行归一化：
[图片]
这里$$x = ( x_1 , x_2 , ⋯   , x_H )$$表示某个时间步LN层的输入向量表示，向量维度为H；h实LN层的输出； g ,b实两个可学习的参数。
为什么层归一化有用？一些解释如下：
1. 减少内部协变量偏移（Internal Covariate Shift）： 内部协变量偏移是指在深度神经网络的训练过程中，每一层输入的分布会发生变化，导致网络的训练变得困难。层归一化通过对每一层的输入进行归一化处理，可以减少内部协变量偏移，使得每一层的输入分布更加稳定。
2. 稳定化梯度： 层归一化有助于保持每一层输出的均值和方差稳定，从而使得梯度的传播更加稳定。这有助于减少梯度消失或梯度爆炸的问题，提高梯度在网络中的流动性，加快训练速度。
3. 更好的参数初始化和学习率调整： 通过层归一化，每一层的输入分布被归一化到均值为0、方差为1的标准正态分布，这有助于更好地初始化网络参数和调整学习率。参数初始化与学习率调整的稳定性对模型的训练效果至关重要。
4. 增强模型的泛化能力： 层归一化可以减少网络对训练数据分布的依赖，降低了过拟合的风险，从而提高模型的泛化能力。稳定的输入分布有助于模型更好地适应不同数据集和任务。
RMSNorm:
虽然LayerNorm很好，但是它每次需要计算均值和方差。RMSNorm的思想就是移除(1)式中$$\mu$$的计算部分：
[图片]
同时在实现也可以移除平移偏置$$b$$。
单看(2)式的话，相当于仅使用$$x$$的均方根来对输入进行归一化，它简化了层归一化的计算，变得更加高效，同时还有可能带来性能上的提升。
实现
RMSNorm的实现很简单：
import torch
import torch.nn as nn
from torch import Tensor

class RMSNorm(nn.Module):
  def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(hidden_size))

  def _norm(self, hidden_states: Tensor) -> Tensor:
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    return hidden_states * torch.rsqrt(variance + self.eps)

  def forward(self, hidden_states: Tensor) -> Tensor:
    return self.weight * self._norm(hidden_states.float()).type_as(hidden_states)
torch.rsqrt是torch.sqrt的倒数；eps是一个很小的数，防止除零；hidden_states.float()确保了标准差计算的精确度和稳定性，然后在forward方法中，通过.type_as(hidden_states)将结果转换回原来的数据类型，以保持与输入张量相同的数据类型，使得归一化处理后的结果与输入数据类型一致。
Pre-Norm or Post-Norm：
[图片]
位置编码：
参数量分析：

激活函数：