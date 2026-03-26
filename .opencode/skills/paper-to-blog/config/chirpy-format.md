# Chirpy Format Configuration

Chirpy 主题格式规范 - 硬编码配置

## Frontmatter 规范

### 必填字段
```yaml
layout: post                    # 必须为 "post"
title: "中文标题"               # 字符串
date: 2024-03-26 15:30:00 +0800 # 格式：YYYY-MM-DD HH:MM:SS +0800
categories: [一级，二级]        # 列表或逗号分隔字符串
tags: [标签 1, 标签 2, ...]      # 列表或逗号分隔字符串，最多 8 个
math: true                      # 布尔值，启用数学公式
mermaid: true                   # 布尔值，启用 Mermaid 图表
image: /assets/images/slug/xxx  # 绝对路径
```

### 字段验证规则

**layout**:
- ✅ 必须为 `post`
- ❌ 不支持其他 layout

**title**:
- ✅ 支持中英文
- ✅ 建议长度：20-60 字符
- ❌ 避免特殊字符：`:` `"` `'` `/`

**date**:
- ✅ 格式：`YYYY-MM-DD HH:MM:SS +0800`
- ❌ 不支持 ISO 8601（`2024-03-26T15:30:00Z`）

**categories**:
- ✅ 必须包含 2 个分类（一级，二级）
- ✅ 使用英文逗号分隔
- ❌ 避免中文逗号（自动修正）
- 示例：`[论文阅读，知识蒸馏]`

**tags**:
- ✅ 数量：5-8 个
- ✅ 使用英文逗号分隔
- ❌ 避免超过 8 个
- 示例：`[LLM, Knowledge Distillation, Chain-of-Thought]`

**image**:
- ✅ 必须是绝对路径（`/assets/images/...`）
- ✅ 文件必须存在
- ❌ 不支持相对路径

---

## 内容结构规范

### 必需章节
1. **论文信息块**（引用格式）
2. **一句话总结**（50-100 字）
3. **背景与动机**
4. **核心方法**（含图片）
5. **实验结果**
6. **AI 分析方法亮点**（三个维度）
7. **总结**
8. **参考链接**

### 图片说明格式

**强制格式**:
```markdown
*图 X：说明文字，解释图片展示的核心概念*
```

**要求**:
- ✅ 必须使用斜体（`*文字*`）
- ✅ 说明核心概念
- ✅ 解释与上下文关联
- ❌ 不要只写"架构图"

**示例**:
```markdown
![框架图](/assets/images/scott-kd/framework.png)
*图 2:SCOTT 框架图，展示对比解码和自洽蒸馏两个核心组件的协作流程*
```

---

## 公式格式规范

### 独立公式
```markdown
$$G(t_i|a^*)=\log\frac{P(t_i|p,q,a^*,t_{<i})}{P(t_i|p,q,a^{'},t_{<i})}$$
```

### 行内公式
```markdown
注意力权重 $\alpha_i$ 的计算...
```

### 禁止格式
- ❌ `\[...\]`
- ❌ `\(...\)`
- ❌ 未闭合的 `$$`

---

## 验证规则

### Frontmatter 检查
1. 所有必填字段存在
2. categories 格式正确（2 个，英文逗号）
3. tags 数量合理（≤8 个）
4. image 路径存在

### 正文检查
1. 所有图片有斜体说明
2. 公式正确闭合
3. 包含"AI 分析方法亮点"章节
4. 包含"参考链接"章节
5. 字数在 1500-5000 之间

### 文件检查
1. image 文件存在
2. 所有引用的图片文件存在

---

## 自动修正规则

| 问题 | 修正方式 |
|------|---------|
| 中文逗号 `,` | 替换为英文逗号 `,` |
| tags 超过 8 个 | 截断为前 8 个 |
| image 相对路径 | 添加 `/assets/images/` 前缀 |
| 标题包含 `:` | 替换为中文冒号 `：` |

---

## 输出模板

```yaml
---
layout: post
title: {title}
date: {date}
categories: [{categories}]
tags: [{tags}]
math: true
mermaid: true
image: {image}
---

> **论文信息**
> - **标题**: {title}
> - **作者**: {authors} ({affiliations})
> - **发布**: {date}
> - **链接**: [arXiv]({arxiv_url})

---

## 一句话总结

{summary}

---

## 背景与动机

{content}

---

## 核心方法

{content}

---

## 实验结果

{content}

---

## AI 分析方法亮点

### 问题定位精准

{content}

### 方法创新

{content}

### 实用性强

{content}

---

## 总结

{content}

---

## 参考链接

1. 论文原文：[链接]({arxiv_url})
2. 代码仓库：[GitHub]({github_url})
```
