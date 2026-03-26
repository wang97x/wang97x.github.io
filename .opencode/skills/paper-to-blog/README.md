# paper-to-blog 快速开始指南

## 安装依赖

```bash
pip install pyyaml
```

## 快速使用

### 从分析结果生成博客

```bash
python .opencode/skills/paper-to-blog/scripts/convert_to_blog.py \
  "_papers/scott-kd/analysis.json" \
  --categories "论文阅读，知识蒸馏" \
  --output "_posts/"
```

### 一键转换（从 arXiv 到博客）

```bash
python .opencode/skills/paper-analyzer/scripts/analyze_paper.py \
  "https://arxiv.org/abs/2305.01879" \
  --categories "论文阅读，知识蒸馏" \
  --next paper-to-blog
```

## 验证生成的博客

```bash
python .opencode/skills/paper-to-blog/scripts/validate_blog.py \
  "_posts/2024-03-26-scott-kd.md"
```

## 输出示例

### Frontmatter

```yaml
---
layout: post
title: AI 记忆综述：从认知理论到 Agent 架构
date: 2024-03-26 15:30:00 +0800
categories: [论文阅读，Agent]
tags: [AI Memory, Agent, 多智能体系统，记忆机制，综述]
math: true
mermaid: true
image: /assets/images/ai-memory-survey/roadmap.png
---
```

### 内容结构

```markdown
> **论文信息**
> - **标题**: Survey on AI Memory
> - **作者**: Ting Bai et al. (BUPT, Huawei)
> - **发布**: 2026/01/15

---

## 一句话总结

本文提出了统一的 AI 记忆理论框架...

---

## 背景与动机

...

---

## 核心方法

![框架图](/assets/images/slug/framework.png)
*图 2：框架图说明*

---

## AI 分析方法亮点

### 问题定位精准
...

### 方法创新
...

### 实用性强
...

---

## 总结
...

---

## 参考链接
1. 论文原文：[链接](url)
```

## 配置选项

### 写作风格

编辑 `config/style-guide.md` 自定义：

- 段落长度
- 语气和语调
- 词汇选择
- 列表使用规范

### Chirpy 格式

编辑 `config/chirpy-format.md` 自定义：

- Frontmatter 字段
- 图片说明格式
- 公式格式
- 验证规则

## 自动修正

以下问题会自动修正：

| 问题 | 修正方式 |
|------|---------|
| 中文逗号 `,` | → 英文逗号 `,` |
| tags 超过 8 个 | → 截断为前 8 个 |
| image 相对路径 | → 添加 `/assets/images/` |

## 验证检查清单

发布前自动检查：

- [ ] Frontmatter 语法正确
- [ ] categories 使用英文逗号
- [ ] tags ≤ 8 个
- [ ] image 文件存在
- [ ] 所有图片有斜体说明
- [ ] 公式正确闭合
- [ ] 包含"AI 分析方法亮点"
- [ ] 包含"参考链接"
- [ ] 字数 1500-5000

## 常见问题

### Q: Frontmatter 验证失败
**A**: 检查 categories 格式，确保使用英文逗号

### Q: 图片说明缺失
**A**: 手动添加斜体说明：`*图 X：说明文字*`

### Q: 标题太长
**A**: 手动修改 `generate_chinese_title()` 函数

### Q: 内容太短/太长
**A**: 手动补充或删减正文明内容

## 与 paper-analyzer 协作

完整工作流：

```
arXiv 链接
    ↓
download_paper.py
    ↓
parse_paper.py → analysis.json
    ↓
extract_figures.py
    ↓
convert_to_blog.py → _posts/*.md
    ↓
validate_blog.py → ✅ 发布
```

## 高级用法

### 自定义标题翻译

编辑 `convert_to_blog.py` 中的 `generate_chinese_title()`：

```python
translations = {
    'AI Memory': 'AI 记忆',
    'Survey': '综述',
    # 添加更多翻译...
}
```

### 自定义分析亮点

编辑 `generate_ai_analysis_highlights()`：

```python
# 添加自定义维度
highlights.append("### 理论深度\n")
highlights.append("论文建立了完整的理论框架...\n")
```

### 自定义验证规则

编辑 `validate_blog.py` 中的 `check_body()`：

```python
# 添加新的检查
if '未来工作' not in body:
    warnings.append("Missing '未来工作' section")
```
