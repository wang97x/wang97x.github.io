# Frontmatter 模板

## 标准格式

```yaml
---
layout: post
title: {中文标题}
date: YYYY-MM-DD HH:MM:SS +0800
categories: [论文阅读，相关领域]
tags: [LLM, 具体技术，其他标签]
math: true
mermaid: true
image: /assets/images/{paper-slug}/pipeline.png
---
```

## 字段说明

### layout
- **固定值**: `post`
- **作用**: 使用博客文章布局

### title
- **格式**: 中文标题，可包含副标题
- **示例**: 
  - `SCOTT：自洽思维链蒸馏 - 让小型模型学会忠实推理`
  - `DC-CoT：数据为中心的思维链蒸馏基准研究`

### date
- **格式**: `YYYY-MM-DD HH:MM:SS +0800`
- **时区**: `+0800` (北京时间)
- **示例**: `2026-03-26 14:30:00 +0800`

### categories
- **格式**: 数组 `[主分类，子分类]`
- **常用分类**:
  - `论文阅读` - 所有论文解读
  - `知识蒸馏` - 蒸馏相关
  - `大语言模型` - LLM 相关
  - `深度学习` - 深度学习基础
  - `自然语言处理` - NLP 相关
- **示例**: `[论文阅读，知识蒸馏]`

### tags
- **格式**: 数组，列出所有相关技术标签
- **常用标签**:
  - `LLM` - 大语言模型
  - `思维链` - Chain of Thought
  - `知识蒸馏` - Knowledge Distillation
  - `推理` - Reasoning
  - `可解释性` - Interpretability
  - `数据增强` - Data Augmentation
  - `泛化能力` - Generalization
- **示例**: `[LLM, 思维链，知识蒸馏，可解释性，推理]`

### math
- **固定值**: `true`
- **作用**: 启用数学公式支持（KaTeX）

### mermaid
- **固定值**: `true`
- **作用**: 启用 Mermaid 图表支持

### image
- **格式**: `/assets/images/{paper-slug}/pipeline.png`
- **作用**: SEO 和社交媒体预览图片
- **建议**: 使用论文的整体框架图

## 完整示例

### 示例 1: SCOTT 论文

```yaml
---
layout: post
title: SCOTT：自洽思维链蒸馏 - 让小型模型学会忠实推理
date: 2026-03-26 14:30:00 +0800
categories: [论文阅读，知识蒸馏]
tags: [LLM, 思维链，知识蒸馏，可解释性，推理]
math: true
mermaid: true
image: /assets/images/SCOTT/pipeline.png
---
```

### 示例 2: DC-CoT 论文

```yaml
---
layout: post
title: DC-CoT：数据为中心的思维链蒸馏基准研究
date: 2026-03-26 10:30:00 +0800
categories: [论文阅读，大语言模型]
tags: [LLM, 思维链，蒸馏，数据增强，泛化能力]
math: true
mermaid: true
image: /assets/images/DC-CoT/fig1_page1.png
---
```

## 文件命名规范

### 博客文章
- **格式**: `YYYY-MM-DD-{paper-slug}.md`
- **示例**: `2026-03-26-SCOTT-自洽思维链蒸馏.md`

### 图片目录
- **格式**: `assets/images/{paper-slug}/`
- **示例**: `assets/images/SCOTT/`

### Paper Slug 生成规则

从论文标题生成：
1. 取主标题首字母缩写或关键词
2. 保持简洁（5-15 个字符）
3. 使用英文或数字

**示例**:
- `SCOTT: Self-Consistent Chain-of-Thought Distillation` → `SCOTT`
- `The Quest for Efficient Reasoning: A Data-Centric Benchmark` → `DC-CoT`
- `Attention Is All You Need` → `Transformer`

## 常见错误

❌ **缺少时区**
```yaml
date: 2026-03-26 14:30:00  # 错误
```

✅ **正确格式**
```yaml
date: 2026-03-26 14:30:00 +0800  # 正确
```

❌ **categories 使用字符串**
```yaml
categories: 论文阅读  # 错误
```

✅ **使用数组**
```yaml
categories: [论文阅读，知识蒸馏]  # 正确
```

❌ **image 使用相对路径**
```yaml
image: assets/images/SCOTT/pipeline.png  # 错误
```

✅ **使用绝对路径**
```yaml
image: /assets/images/SCOTT/pipeline.png  # 正确
```

## 自定义字段（可选）

### published
- **类型**: boolean
- **默认**: `true`
- **用途**: 控制是否发布

### author
- **类型**: string
- **默认**: 使用 site 配置中的作者
- **用途**: 指定特定文章的作者

### excerpt
- **类型**: string
- **用途**: 自定义摘要（用于 SEO）

```yaml
---
layout: post
title: 标题
excerpt: 这是一篇关于 SCOTT 方法的详细解读，介绍了如何通过对比解码和反事实推理训练忠实的学生模型。
---
```

## SEO 优化建议

1. **title**: 包含关键词，长度 50-60 字符
2. **image**: 使用清晰的框架图，尺寸 1200x630 为佳
3. **tags**: 5-8 个相关标签
4. **categories**: 2 个层级分明的分类
