# Blog Writing Workflow

This repository keeps the blog generation workflow under `tools/` so the content directory stays focused on real posts.

## Files

- `tools/new-post.ps1`: main scaffold generator
- `tools/MDCreating.bat`: Windows interactive wrapper
- `tools/blog-writing-workflow.md`: workflow and usage guide

## Purpose

Use this workflow when you want to turn notes into a post under `_posts/`.

The default structure is:

1. Front matter
2. One-sentence summary
3. Background and problem
4. Core concepts and principles
5. Key details or examples
6. Practical tips and caveats
7. Summary

## Front Matter

Default fields:

```yaml
---
layout: post
title: "Post Title"
author: "wang"
date: 2026-06-23 12:00:00 +0800
categories: [检索与排序]
tags: [概念笔记]
---
```

Use exactly one topic category by default. The current top-level categories are:

- `AI Agent`
- `大模型与训练`
- `检索与排序`
- `工具与框架`
- `编程基础`
- `博客建设`

Keep writing form and framework names in `tags`, not `categories`. Common examples:

- Content form: `论文阅读`, `源码拆解`, `概念笔记`, `工程实践`
- Frameworks and tools: `LangChain`, `LangGraph`, `SentenceTransformers`, `FAISS`, `Gradio`

Optional fields:

- `math: true`
- `mermaid: true`
- `image: /assets/images/...`

## Usage

Create a post scaffold directly:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\new-post.ps1 -Title "Vector Retrieval Basics" -Categories "检索与排序" -Tags "Retriever","概念笔记"
```

If needed, add:

```powershell
-Math -Mermaid
```

For a header image:

```powershell
-Image "/assets/images/vector-retrieval/cover.png"
```

Use the interactive wrapper on Windows:

```bat
.\tools\MDCreating.bat
```

## Classification Rules

- `categories` only express the reader-facing topic.
- Keep most posts to one category.
- Move writing form, frameworks, models, and libraries into `tags`.
- Prefer normalized tags such as `LLM`, `LangChain`, `DeepSeek`, and `FAISS`.

## Notes

- Generated posts are written to `_posts/`.
- Tooling files should stay under `tools/`.
- `_posts/` should only contain blog posts and placeholders.
