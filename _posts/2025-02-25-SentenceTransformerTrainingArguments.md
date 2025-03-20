---
layout: post
title: "SentenceTransformerTrainingArguments"
author: "wang"
date: 2025-02-25 09:59:00 +0800
categories: [Information Retrieval]
tags: [SentenceTransformers, SentenceTransformerTrainingArguments]
---
`SentenceTransformerTrainingArguments` 是用于配置 Sentence Transformers 模型训练的参数类，继承自 `TrainingArguments` 并添加了一些特定于 Sentence Transformers 的参数。以下是其主要参数的解析：

### 训练和评估相关参数

| 参数                            | 作用                                                      |
|-------------------------------|---------------------------------------------------------|
| `output_dir`                  | 模型检查点的输出目录。                                             |
| `overwrite_output_dir`        | 是否覆盖输出目录中的现有内容。                                         |
| `do_train`                    | 是否执行训练过程。                                               |
| `do_eval`                     | 是否执行评估过程。                                               |
| `eval_strategy`               | 评估策略，可选值为 `"no"`（不评估）、`"steps"`（按步评估）或 `"epoch"`（按轮评估）。 |
| `eval_steps`                  | 若 `eval_strategy` 为 `"steps"`，则每多少步进行一次评估。              |
| `per_device_train_batch_size` | 每个设备（GPU/TPU/CPU）的训练批次大小。                               |
| `per_device_eval_batch_size`  | 每个设备的评估批次大小。                                            |
| `gradient_accumulation_steps` | 梯度累积步数，用于模拟更大的批次大小。                                     |
| `num_train_epochs`            | 总训练轮数。                                                  |
| `max_steps`                   | 总训练步数，若设置为正数，则会覆盖 `num_train_epochs`。                   |

### 优化器和学习率调度器相关参数

| 参数                              | 作用                                  |
|---------------------------------|-------------------------------------|
| `learning_rate`                 | 初始学习率，默认为 `5e-5`。                   |
| `weight_decay`                  | 权重衰减系数，默认为 `0.0`。                   |
| `adam_beta1` 和 `adam_beta2`     | Adam 优化器的超参数，默认分别为 `0.9` 和 `0.999`。 |
| `adam_epsilon`                  | Adam 优化器的 epsilon 值，默认为 `1e-8`。     |
| `lr_scheduler_type`             | 学习率调度器类型，默认为 `"linear"`。            |
| `warmup_ratio` 和 `warmup_steps` | 线性预热的比例或步数，用于学习率预热。                 |

### 保存和日志相关参数

| 参数                 | 作用                              |
|--------------------|---------------------------------|
| `save_strategy`    | 模型保存策略，可选值与 `eval_strategy` 相同。 |
| `save_steps`       | 若保存策略为 `"steps"`，则每多少步保存一次模型。   |
| `save_total_limit` | 最多保存的模型数量。                      |
| `logging_strategy` | 日志记录策略，可选值与 `eval_strategy` 相同。 |
| `logging_steps`    | 若日志策略为 `"steps"`，则每多少步记录一次日志。   |
| `logging_dir`      | 日志文件的存储目录。                      |

### 硬件和性能相关参数

| 参数                       | 作用                         |
|--------------------------|----------------------------|
| `no_cuda`                | 是否禁用 GPU。                  |
| `use_cpu`                | 是否强制使用 CPU。                |
| `fp16` 和 `bf16`          | 是否使用半精度（FP16）或脑浮点（BF16）训练。 |
| `gradient_checkpointing` | 是否启用梯度检查点以节省内存。            |

### 数据采样和多数据集相关参数

| 参数                            | 作用                                                    |
|-------------------------------|-------------------------------------------------------|
| `batch_sampler`               | 批次采样器的类型，例如 `BatchSamplers.NO_DUPLICATES`。            |
| `multi_dataset_batch_sampler` | 多数据集采样策略，例如 `MultiDatasetBatchSamplers.PROPORTIONAL`。 |

### 其他参数

| 参数                       | 作用                         |
|--------------------------|----------------------------|
| `seed`                   | 随机种子，用于保证实验的可重复性。          |
| `push_to_hub`            | 是否将模型推送到 Hugging Face Hub。 |
| `resume_from_checkpoint` | 是否从检查点恢复训练。                |

这些参数提供了灵活的配置选项，以适应不同的训练需求和硬件环境。
