---
layout: post
title: "CoT-Self-Instruct​​：高质量数据合成"
author: "wang"
date: 2025-08-12 17:12:00 +0800
categories: [论文阅读]
tags: [数据合成]
---
COT-SELF-INSTRUCT: BUILDING HIGH-QUALITY SYNTHETIC PROMPTS FOR REASONING AND NON-REASONING TASKS

Paper：[https://arxiv.org/pdf/2507.23751](https://arxiv.org/pdf/2507.23751)

## 1. 摘要（ABSTRACT）

- 核心方法：提出 CoT-Self-Instruct，一种通过思维链（Chain-of-Thought, CoT）引导大语言模型（LLM）生成高质量合成指令的框架。

- 创新点：
  - 推理任务：首先生成带逐步推理的指令，再通过答案一致性过滤（Answer-Consistency） 筛选数据。
  - 非推理任务：使用拒绝指令偏好（RIP） 过滤低质量指令。

- 成果：
  - 在 MATH500、AMC23 等推理任务中，性能显著超越现有数据集（如s1k、OpenMathReasoning）。
  - 在 AlpacaEval 2.0、Arena-Hard 非推理任务中，超越人类标注数据和传统自指令方法。

## 2. 引言（INTRODUCTION）
- 背景：
    - 高质量训练数据稀缺且成本高，人类数据存在偏差问题。
    - 现有自指令方法（如Self-Instruct）难以保证合成数据的质量和复杂性。
- 问题定义：
  - 传统方法直接生成指令，缺乏规划步骤，导致数据质量不稳定。
- 解决方案：
  - CoT-Self-Instruct引入推理规划阶段，让LLM分析种子指令的属性（领域、复杂度），再生成新指令。

## 3. 相关工作（RELATED WORK）
- 合成数据生成：
  - Self-Instruct：利用种子指令引导LLM生成新数据。
  - Evol-Instruct：通过重写增加指令复杂度。
  - 其他方法：多跳问答、工具使用等场景的合成数据生成。
- 数据筛选方法：
  - 传统方法：去重、聚类（如ROUGE-L）。
  - 新方法：LLM作为质量评判器、偏好优化（如RIP、Self-Consistency）。

## 4. 方法（COT-SELF-INSTRUCT）
### 4.1 指令生成（Synthetic Instruction Creation）
- 推理任务：  
  LLM生成指令+可验证答案，要求答案格式明确（如数值/选项）。  
  
- 非推理任务：  
  LLM仅生成开放式指令（如写作、编程），响应质量通过奖励模型评估。  

### 4.2 指令筛选（Synthetic Instruction Curation）
- 推理任务：  
  Answer-Consistency过滤：若LLM多次生成的答案与CoT生成的目标答案不一致，则丢弃该指令。
- 非推理任务：  
  RIP过滤：基于奖励模型分数分布，保留高分指令（如最低分≥50%分位数）。

### 4.3 自训练（Self-training）
- 推理任务：使用GRPO（基于可验证奖励的强化学习）。
- 非推理任务：采用DPO（直接偏好优化），结合长度归一化避免响应膨胀。

## 5. 实验（EXPERIMENTAL RESULTS）

### 5.1 推理任务
- 数据集：s1k种子指令 → 生成5000条合成指令。
- 关键结果：
  - CoT-Self-Instruct（53.0%）> Self-Instruct（49.5%）。
  - 过滤后：Answer-Consistency进一步提升至57.2%。
  - 对比基线：超越s1k（44.6%）和OpenMathReasoning（47.5%）。

### 5.2 非推理任务
- 数据集：WildChat种子指令 → 按8类领域生成指令。
- 关键结果：
  - CoT-Self-Instruct（53.9%）> Self-Instruct（47.4%）。
  - RIP过滤后：性能达54.7%（表2）。
  - 在线DPO训练：最高达67.1%，显著超越人类数据（63.1%）。

## 6. 结论（CONCLUSION）
- 贡献：  
  CoT-Self-Instruct通过推理规划+严格过滤，在合成数据质量上实现突破。
- 影响：  
  为LLM训练提供高效、低成本的数据生成方案，适用于复杂推理和开放指令任务。

## 附录（APPENDIX）
- 消融实验：
  - 推理任务：验证不同模板效果，证明CoT生成优于直接生成。
  - 非推理任务：长链CoT显著优于短链。
- 规模控制：  
  在相同训练量下（893条），CoT方法仍领先基线。

总结：该论文通过将思维链引入自指令框架，结合针对性过滤机制，显著提升了合成数据的质量和模型性能，为LLM训练提供了新范式。