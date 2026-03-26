---
name: paper-to-blog
description: 将 arXiv 论文链接转化为适合发布的博客文章。使用场景：用户提供 arXiv 论文 URL 时，自动下载论文、提取关键图表、转化为博客风格 Markdown，并按照项目规范存储到_posts/和 assets/images/目录。支持完整工作流程：下载→解析→提取→转换→发布。
allowed-tools: Read, Write, Bash
---

# Paper to Blog Converter

将 arXiv 论文自动转换为符合 Chirpy 主题规范的博客文章。

## 触发条件

当用户提供：
- arXiv 链接（如 `https://arxiv.org/abs/2305.01879`）
- 论文 PDF 文件
- 论文标题 + 要求转化为博客

时触发此技能。

## 完整工作流程

### 步骤 1: 解析 arXiv ID

从用户输入提取 arXiv ID：
- `https://arxiv.org/abs/2305.01879` → `2305.01879`
- `https://arxiv.org/pdf/2305.01879.pdf` → `2305.01879`
- `arXiv:2305.01879` → `2305.01879`

### 步骤 2: 下载论文

使用 `download_paper.py` 脚本：

```bash
python ".opencode/skills/paper-to-blog/scripts/download_paper.py" \
  "{arxiv_id}" \
  "papers/{arxiv_id}"
```

**下载内容**:
- PDF: `https://arxiv.org/pdf/{arxiv_id}`
- 源码：`https://arxiv.org/e-print/{arxiv_id}`

### 步骤 3: 解析论文内容

使用 `parse_latex.py` 脚本分析 LaTeX 源码：

```bash
python ".opencode/skills/paper-to-blog/scripts/parse_latex.py" \
  "papers/{arxiv_id}"
```

**提取内容**:
- 论文标题、作者、机构
- 摘要（Abstract）
- 各章节内容（Introduction, Method, Experiments, Conclusion）

### 步骤 4: 提取关键图表

使用 `extract_figures.py` 脚本：

```bash
python ".opencode/skills/paper-to-blog/scripts/extract_figures.py" \
  "papers/{arxiv_id}" \
  "assets/images/{paper-slug}"
```

**图片选择策略**（最多 8 张）:

**必选**:
- ✅ 架构图/流程图（pipeline.pdf, framework.pdf）
- ✅ 核心方法示意图（method.pdf, approach.pdf）

**可选**:
- ✅ 主要实验结果图（acc_las.pdf, results.pdf）
- ✅ 关键消融实验图（ablation.pdf, student_size.pdf）

**排除**:
- ❌ Logo/装饰性图片
- ❌ 纯表格（用文字描述）
- ❌ 重复/相似图片

**优先级顺序**:
1. pipeline / framework / architecture
2. method / approach / overview
3. contrastive / counterfactual / training
4. teacher / student / acc_las
5. perf_change / ablation / student_size

### 步骤 5: 生成博客文章

#### 5.1 创建文件

```bash
# 博客文件
_posts/YYYY-MM-DD-{paper-slug}-{中文标题}.md

# 图片目录
assets/images/{paper-slug}/
```

#### 5.2 Frontmatter 格式

```yaml
---
layout: post
title: {中文标题}
date: YYYY-MM-DD HH:MM:SS +0800
categories: [论文阅读，相关领域]
tags: [LLM, 具体技术]  # 使用英文逗号分隔多个标签
math: true
mermaid: true
image: /assets/images/{paper-slug}/pipeline.png
---
```

#### 5.3 内容结构

```markdown
> **论文信息**
> - **标题**: 英文标题
> - **作者**: Author et al. (机构)
> - **会议**: 会议/期刊
> - **arXiv**: [链接](url)
> - **代码**: [GitHub](url) (如公开)

---

## 一句话总结

[核心贡献]

---

## 背景与动机

[问题重要性 + 现有方法局限]

---

## 核心方法

### 整体框架

![框架图](/assets/images/{paper-slug}/pipeline.png)
*图 1：框架说明*

### 关键技术 1

[技术细节 + 公式]

![方法图](/assets/images/{paper-slug}/method.png)
*图 2：方法说明*

---

## 实验结果

### 主要结果

![结果图](/assets/images/{paper-slug}/results.png)
*图 X：结果说明*

**发现**:
- 关键发现

---

## AI 分析方法亮点

**写作要求**：整合为三个维度，删除"应用价值"和"局限性"子标题

1. **问题定位精准**: [直接针对论文解决的核心问题]
2. **方法创新**: [技术/方法上的创新点，包含具体数据]
3. **实用性强**: [实际应用价值或优势，包含性能对比]

---

## 总结

[核心 takeaway]

---

## 参考链接
```

### 步骤 6: 清理临时文件

```bash
rm -rf "papers/{arxiv_id}"
```

## 写作风格指南

### 平衡型风格

**保留**:
- ✅ 关键公式和技术细节
- ✅ 核心实验数据
- ✅ 方法创新点

**简化**:
- ✅ 复杂数学推导
- ✅ 过多实验细节
- ✅ 学术化表述

**添加**:
- ✅ 通俗解释
- ✅ 实际应用场景
- ✅ AI 分析方法亮点（三个维度）

### 图片说明格式

```markdown
![图片标题](/assets/images/{paper-slug}/xxx.png)
*图 X：说明文字，解释图片展示的核心概念*
```

**要点**:
- 使用斜体 `*文字*`
- 说明图片展示的核心概念
- 解释与上下文的关联

### 数学公式格式

使用 `$$` 包裹 LaTeX 公式：

```markdown
$$G(t_i|a^*)=\log\frac{P(t_i|p,q,a^*,t_{<i})}{P(t_i|p,q,a^{'},t_{<i})}$$
```

### 字数建议

| 部分 | 建议字数 |
|------|----------|
| 一句话总结 | 50-100 字 |
| 背景与动机 | 300-500 字 |
| 核心方法 | 800-1200 字 |
| 实验结果 | 400-600 字 |
| 个人见解 | 300-500 字 |
| **总计** | **2000-3000 字** |

## 项目规范

### 目录结构

```
_posts/
└── YYYY-MM-DD-{paper-slug}-{中文标题}.md

assets/images/{paper-slug}/
├── pipeline.png
├── method.png
├── results.png
└── ...
```

### 文件命名

**博客文章**: `YYYY-MM-DD-{paper-slug}-{中文标题}.md`
- 日期格式：`YYYY-MM-DD`
- Paper slug: 英文缩写（5-15 字符）
- 中文标题：简短描述

**图片目录**: `assets/images/{paper-slug}/`
- 使用绝对路径：`/assets/images/{paper-slug}/xxx.png`

### 分类和标签

**categories**: `[论文阅读，具体领域]`
- 第一级：`论文阅读`
- 第二级：`知识蒸馏`, `大语言模型`, `深度学习`, `自然语言处理`, `数据合成` 等
- **注意**: 使用英文逗号 `,` 分隔分类标签

**tags**: `[LLM, 具体技术, ...]`
- 5-8 个相关标签
- 示例：`[LLM, 思维链, 知识蒸馏, 可解释性, 推理]`
- **注意**: 使用英文逗号 `,` 分隔标签

## 依赖项

- Python 3.x
- PyMuPDF (fitz) - PDF 转 PNG
- curl - 下载文件
- tar - 解压源码包

## 错误处理

### 常见问题

**1. 下载失败**
```
解决方案：检查网络连接，重试下载
```

**2. 找不到 figures 目录**
```
解决方案：从 PDF 直接提取图片（备选方案）
```

**3. 图片转换失败**
```
解决方案：降低 zoom 参数或跳过该图片
```

**4. LaTeX 解析错误**
```
解决方案：手动读取 main.tex 和 sections/*.tex
```

## 示例

### 输入

```
https://arxiv.org/abs/2305.01879 分类：论文阅读，知识蒸馏
```

### 输出

```
_posts/2026-03-26-SCOTT-自洽思维链蒸馏.md (8.1KB)
assets/images/SCOTT/
├── pipeline.png
├── contrastive_decoding.png
├── counterfactual_training.png
├── teacher_las.png
├── acc_las_csqa.png
├── student_size_CSQA.png
├── perf_change_csqa.png
└── vacuous_rationale.png
```

## 参考文档

- **写作风格**: 详见 `references/blog-style-guide.md`
- **Frontmatter**: 详见 `references/frontmatter-template.md`
- **文章模板**: 详见 `assets/templates/blog-post-template.md`

## 检查清单

发布前检查：
- [ ] Frontmatter 格式正确（categories、tags 使用英文逗号分隔）
- [ ] 所有图片都有详细说明（斜体格式）
- [ ] 公式使用 `$$` 包裹
- [ ] 包含"AI 分析方法亮点"部分（三个维度：问题定位精准、方法创新、实用性强）
- [ ] 参考链接完整（只保留公开的链接）
- [ ] 分类和标签正确（categories 二级分类，tags 5-8 个）
- [ ] 图片路径使用绝对路径 `/assets/images/...`
- [ ] 临时文件已清理
