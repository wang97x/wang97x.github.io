---
name: paper-to-blog
description: 将论文分析结果转换为符合 Chirpy 主题规范的博客文章。Use when: (1) paper-analyzer 已生成 analysis.json 需要转换为博客，(2) 用户提供结构化论文数据需要生成博客，(3) 需要将论文笔记发布到 Jekyll/Chirpy 博客，(4) 需要确保 Frontmatter 和图片格式符合 Chirpy 规范
---

# Paper to Blog Converter

将论文分析结果自动转换为符合 Chirpy 主题规范的博客文章。

## 核心职责

读取 paper-analyzer 生成的 `analysis.json`，输出：
- `_posts/YYYY-MM-DD-slug-title.md`（符合 Chirpy Frontmatter）
- `assets/images/{slug}/`（重命名的图片 + 说明）

## 硬编码规范（Chirpy 强制要求）

以下规范必须严格遵守，由转换器自动保证：

### Frontmatter 格式

```yaml
---
layout: post
title: "{中文标题}"
date: {YYYY-MM-DD HH:MM:SS +0800}
categories: [{一级分类},{二级分类}]  # 英文逗号分隔
tags: [标签 1, 标签 2, ..., 标签 N]  # 英文逗号分隔，5-8 个
math: true
mermaid: true
image: /assets/images/{slug}/{封面图}
---
```

**验证规则**:
- ✅ `categories` 必须使用英文逗号（`,`）而非中文逗号（`，`）
- ✅ `tags` 必须使用英文逗号（`,`）而非中文逗号（`，`）
- ✅ `date` 格式必须为 `YYYY-MM-DD HH:MM:SS +0800`
- ✅ `image` 路径必须是绝对路径（`/assets/images/...`）

### 内容结构模板

```markdown
> **论文信息**
> - **标题**: {英文标题}
> - **作者**: {Author et al.} ({机构})
> - **会议**: {会议/期刊}
> - **arXiv**: [链接](url)
> - **代码**: [GitHub](url) (如公开)

---

## 一句话总结

{50-100 字核心贡献总结}

---

## 背景与动机

{问题重要性 + 现有方法局限}

---

## 核心方法

### 整体框架

![框架图](/assets/images/{slug}/{figure}.png)
*图 1：框架说明（斜体格式）*

### 关键技术 1

{技术细节 + 公式}

---

## 实验结果

### 主要结果

![结果图](/assets/images/{slug}/{figure}.png)
*图 X：结果说明*

**发现**:
- 关键发现 1
- 关键发现 2

---

## AI 分析方法亮点

1. **问题定位精准**: {直接针对论文解决的核心问题}
2. **方法创新**: {技术/方法上的创新点，包含具体数据}
3. **实用性强**: {实际应用价值或优势，包含性能对比}

---

## 总结

{核心 takeaway}

---

## 参考链接

1. 论文原文：[链接](url)
2. 代码仓库：[链接](url)
3. 项目主页：[链接](url)
```

### 图片说明格式

**强制格式**（斜体）:
```markdown
*图 X：说明文字，解释图片展示的核心概念*
```

**要求**:
- ✅ 必须使用斜体（`*文字*`）
- ✅ 说明图片展示的核心概念
- ✅ 解释与上下文的关联
- ❌ 不要只写"架构图"、"结果图"等简短描述

### 数学公式格式

使用 `$$` 包裹 LaTeX 公式：
```markdown
$$G(t_i|a^*)=\log\frac{P(t_i|p,q,a^*,t_{<i})}{P(t_i|p,q,a^{'},t_{<i})}$$
```

**要求**:
- ✅ 独立公式用 `$$...$$`
- ✅ 行内公式用 `$...$`
- ❌ 不要使用 `\[...\]` 或 `\(...\)`

## 可配置部分（写作风格）

以下风格配置从 `config/style-guide.md` 读取，允许灵活调整：

### 段落组织

- **段落长度**: 2-4 句为佳
- **小标题频率**: 每 2-3 段一个 `###` 标题
- **字数范围**: 2000-3000 字（总计）

### 语气和语调

- **直接**: 陈述观点明确，不模棱两可
- **观点鲜明**: 允许表达 contrarian 观点
- **第一人称**: 分享经验时使用"我"
- **过渡词**: 使用"that said", "however" 等

### 词汇选择

**推荐**:
- leverage, key, that said, directly

**避免**:
- utilize, important, very, really

### 列表使用

- 避免过多 bullet points
- 优先使用连贯的段落叙述
- 对比/总结时使用表格

## 工作流程

### 1. 读取分析 JSON

```python
import json

with open("_papers/{slug}/analysis.json", "r", encoding="utf-8") as f:
    analysis = json.load(f)
```

### 2. 生成博客草稿

```bash
python ".opencode/skills/paper-to-blog/scripts/convert_to_blog.py" \
  "_papers/{slug}/analysis.json" \
  --categories "论文阅读，知识蒸馏" \
  --output "_posts/"
```

**核心步骤**:
1. 生成 Frontmatter（验证格式）
2. 生成论文信息块
3. 生成"一句话总结"
4. 生成各章节内容（背景、方法、实验）
5. 生成"AI 分析方法亮点"（三个维度）
6. 生成总结和参考链接

### 3. 图片处理

```bash
python ".opencode/skills/paper-to-blog/scripts/rename_images.py" \
  "_papers/{slug}/figures/" \
  "assets/images/{slug}/" \
  --analysis "_papers/{slug}/analysis.json"
```

**重命名规则**:
- `fig_1_1.png` → `roadmap.png`（第 1 章第 1 张，类型为 overview）
- `fig_3_1.png` → `method.png`（第 3 章第 1 张，类型为 method）
- `fig_5_2.png` → `results.png`（第 5 章第 2 张，类型为 result）

**自动添加说明**:
```markdown
![框架图](/assets/images/{slug}/method.png)
*图 2：方法示意图，展示核心算法流程*
```

### 4. 发布前检查

```bash
python ".opencode/skills/paper-to-blog/scripts/validate_blog.py" \
  "_posts/YYYY-MM-DD-slug-title.md"
```

**检查清单**:
- [ ] Frontmatter 语法正确（YAML 格式）
- [ ] categories/tags 使用英文逗号
- [ ] 所有图片都有斜体说明
- [ ] 图片路径存在（文件检查）
- [ ] 公式使用 `$$` 包裹
- [ ] 包含"AI 分析方法亮点"部分
- [ ] 参考链接完整

### 5. 清理临时文件（可选）

```bash
# 询问用户
是否保留分析文件？
- 保留（便于后续修改）
- 删除（仅保留博客和图片）
```

## 输出位置

```
_posts/
└── YYYY-MM-DD-slug-title.md

assets/images/{slug}/
├── roadmap.png
├── architecture.png
├── method.png
├── results.png
└── ablation.png
```

## 错误处理

| 错误 | 处理方式 |
|------|---------|
| analysis.json 不存在 | 终止，提示先运行 paper-analyzer |
| 图片文件缺失 | 警告，跳过该图片继续生成 |
| Frontmatter 验证失败 | 终止，报告具体错误 |
| 分类标签格式错误 | 自动修正（中文逗号→英文逗号） |

## 与 paper-analyzer 的协作

**标准流程**:
```
用户输入 arXiv 链接
    ↓
paper-analyzer 下载 + 解析 → _papers/slug/analysis.json
    ↓
paper-to-blog 读取 JSON → _posts/slug-title.md
    ↓
询问是否清理 _papers/slug/
```

**一键调用**:
```bash
python ".opencode/skills/paper-to-blog/scripts/convert_pipeline.py" \
  "https://arxiv.org/abs/2305.01879" \
  --categories "论文阅读，知识蒸馏" \
  --auto-cleanup
```

## 示例

### 输入
```json
{
  "metadata": {
    "title": "Survey on AI Memory",
    "authors": ["Ting Bai", "Jiayang Fan"],
    "affiliations": ["BUPT", "Huawei"],
    "date": "2026-01-15",
    "links": {
      "github": "https://github.com/BAI-LAB/Survey-on-AI-Memory",
      "homepage": "https://baijia.online/homepage/memory_survey.html"
    }
  },
  "key_contributions": ["提出 4W 记忆分类法", "首个系统性评估框架"],
  "tags": ["AI Memory", "Agent", "Survey"],
  "figures": [
    {"path": "fig_1_1.png", "suggested_name": "roadmap.png", "type": "overview"}
  ]
}
```

### 输出 Frontmatter
```yaml
---
layout: post
title: AI 记忆综述：从认知理论到 Agent 架构
date: 2026-03-26 16:00:00 +0800
categories: [论文阅读，Agent]
tags: [AI Memory, Agent, 多智能体系统，记忆机制，综述，认知科学]
math: true
mermaid: true
image: /assets/images/ai-memory-survey/roadmap.png
---
```

### 输出正文片段
```markdown
> **论文信息**
> - **标题**: Survey on AI Memory: Theories, Taxonomies, Evaluations, and Emerging Trends
> - **作者**: Ting Bai, Jiayang Fan et al. (BUPT, Huawei)
> - **发布**: 2026/01/15
> - **项目主页**: [链接](https://baijia.online/homepage/memory_survey.html)
> - **GitHub**: [链接](https://github.com/BAI-LAB/Survey-on-AI-Memory)

---

## 一句话总结

本文提出了统一的 AI 记忆理论框架，通过独创的 4W 记忆分类法系统梳理了单智能体和多智能体系统中的记忆机制。

---

## 核心方法

### 4W 记忆分类法

![4W 记忆分类法](/assets/images/ai-memory-survey/4w-taxonomy.png)
*图 3:4W 记忆分类法，包含四个正交维度：When（生命周期）、What（记忆类型）、How（存储形式）、Which（模态类型）*
```

## 参考文档

- **Chirpy 主题文档**: https://github.com/cotes2020/jekyll-theme-chirpy
- **写作风格指南**: 详见 `config/style-guide.md`
- **Frontmatter 模板**: 详见 `config/frontmatter-template.md`
- **验证规则**: 详见 `config/validation-rules.json`
