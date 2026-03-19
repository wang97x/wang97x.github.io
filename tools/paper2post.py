#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


@dataclass(frozen=True)
class PaperInfo:
    title: str
    authors: str
    venue: str
    year: str
    doi: str
    keywords: list[str]
    abstract: str
    contributions: list[str]
    conclusion: str
    datasets: list[str]
    model_name: str


def _read_pdf_text(pdf_path: Path) -> tuple[str, dict[str, str]]:
    reader = PdfReader(str(pdf_path))
    meta: dict[str, str] = {}
    if reader.metadata:
        for k, v in reader.metadata.items():
            if v is None:
                continue
            meta[str(k)] = str(v)

    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    text = "\n".join(pages)
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, meta


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _clean_inline(s: str) -> str:
    s = s.replace("\u00ad", "")  # soft hyphen
    s = s.replace("\ufb01", "fi").replace("\ufb02", "fl")
    s = re.sub(r"-\n([a-z])", r"\1", s)  # de-hyphenate line breaks
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _first_match(patterns: Iterable[str], text: str, flags: int = 0) -> re.Match[str] | None:
    for p in patterns:
        m = re.search(p, text, flags=flags)
        if m:
            return m
    return None


def _extract_between(text: str, start_patterns: list[str], end_patterns: list[str], flags: int = 0) -> str:
    start = _first_match(start_patterns, text, flags=flags)
    if not start:
        return ""
    tail = text[start.end() :]
    end = _first_match(end_patterns, tail, flags=flags)
    chunk = tail[: end.start()] if end else tail[:5000]
    return _clean_inline(chunk)


def _extract_contributions(text: str) -> list[str]:
    m = _first_match(
        [
            r"The main contributions are summarized as follows:(.*?)(?:The rest of the paper is organized as follows\.|Section\s+\d)",
            r"Our contributions are summarized as follows:(.*?)(?:The rest of the paper is organized as follows\.|Section\s+\d)",
        ],
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return []
    body = m.group(1)
    body = body.replace("\n", " ")
    parts = re.split(r"[•\u2022]\s*", body)
    items = []
    for p in parts:
        p = _clean_inline(p)
        if not p:
            continue
        items.append(p)
    return items[:10]


def _extract_conclusion(text: str) -> str:
    block = _extract_between(
        text,
        [r"\n7\.\s*Conclusion\b", r"\nConclusion\b"],
        [r"\nCRediT authorship contribution statement\b", r"\nReferences\b"],
        flags=re.IGNORECASE,
    )
    return block


def _guess_model_name(text: str) -> str:
    m = re.search(r"\b([A-Z]{2,}(?:-[A-Z0-9]{1,})+)\b", text)
    if m:
        return m.group(1)
    return ""


def _extract_datasets(text: str) -> list[str]:
    candidates = set()
    for name in ["Math23K", "MAWPS", "SVAMP", "MathQA", "ASDiv", "GSM8K", "APE210K"]:
        if re.search(rf"\b{re.escape(name)}\b", text):
            candidates.add(name)
    # try a generic dataset line
    m = re.search(r"\bDatasets?\b(.*?)(?:\n\d+\.\s|\nReferences\b)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        window = m.group(1)[:2000]
        for tok in re.findall(r"\b[A-Za-z][A-Za-z0-9\-]{2,}\b", window):
            if tok.lower() in {"dataset", "datasets", "table", "fig", "section"}:
                continue
            if tok in {"Math23K", "MAWPS"}:
                candidates.add(tok)
    return sorted(candidates)


def _slugify(title: str) -> str:
    t = title.lower()
    t = re.sub(r"[^a-z0-9]+", "-", t)
    t = t.strip("-")
    t = re.sub(r"-{2,}", "-", t)
    return t[:80] if t else "paper"


def _extract_info(text: str, meta: dict[str, str]) -> PaperInfo:
    title = meta.get("/Title", "").strip() or ""
    authors = meta.get("/Author", "").strip() or ""
    doi = meta.get("/doi", "").strip() or ""
    keywords_raw = meta.get("/Keywords", "").strip()
    keywords = [k.strip() for k in re.split(r"[;,]\s*|\s{2,}", keywords_raw) if k.strip()] if keywords_raw else []
    venue = ""
    year = ""
    subject = meta.get("/Subject", "")
    m = re.search(r"([A-Za-z][A-Za-z ]+),\s*(\d{4})", subject)
    if m:
        venue = m.group(1).strip()
        year = m.group(2)
    if not year:
        m2 = re.search(r"\b(19|20)\d{2}\b", text[:800])
        if m2:
            year = m2.group(0)

    abstract = _extract_between(
        text,
        [r"\bA\s*B\s*S\s*T\s*R\s*A\s*C\s*T\b", r"\bAbstract\b"],
        [r"\n1\.\s*Introduction\b", r"\nIntroduction\b"],
        flags=re.IGNORECASE,
    )
    contributions = _extract_contributions(text)
    conclusion = _extract_conclusion(text)
    datasets = _extract_datasets(text)

    model_name = _guess_model_name(abstract) or _guess_model_name(text)

    if not title:
        # try to infer title from the first page
        first = text[:1200]
        lines = [ln.strip() for ln in first.split("\n") if ln.strip()]
        # pick the longest line that looks like a title
        best = ""
        for ln in lines[:40]:
            if len(ln) < 12 or len(ln) > 140:
                continue
            if any(bad in ln for bad in ["Neurocomputing", "Contents lists available"]):
                continue
            if len(ln) > len(best) and re.search(r"[A-Za-z]", ln):
                best = ln
        title = best or "Untitled paper"

    return PaperInfo(
        title=title,
        authors=authors,
        venue=venue,
        year=year,
        doi=doi,
        keywords=keywords,
        abstract=abstract,
        contributions=contributions,
        conclusion=conclusion,
        datasets=datasets,
        model_name=model_name,
    )


def _mermaid_flow(model_name: str) -> str:
    label = model_name or "Model"
    return f"""```mermaid
flowchart LR
  T[MWP 文本] --> SDP[语义依存解析]
  SDP --> SG[语义图 (Semantic Graph)]
  T --> Q[数量/量词抽取]
  Q --> QG[数量图 (Quantitative Graph)]
  SG --> ENC[多视图图编码器\\n(重构 + 对齐)]
  QG --> ENC
  ENC --> DEC[树结构解码器\\n(Graph2Tree)]
  DEC --> EQ[方程/表达式]
  EQ --> A[答案]
  ENC -->|输出表示| R[{label}]
```"""


def _mermaid_mindmap(model_name: str, datasets: list[str]) -> str:
    root = model_name or "MWP Solver"
    ds = "\n      ".join(f"({d})" for d in datasets) if datasets else "(Math23K/MAWPS)"
    return f"""```mermaid
mindmap
  root(({root}))
    任务
      "数学应用题 (MWP)"
      "生成解题表达式"
    图构建
      "语义依存图"
      "数量关系图"
    学习策略
      "结构重构 (adaptive)"
      "跨视图对齐 (consistent)"
    解码
      "树解码 (Graph2Tree)"
      "长度归一化损失"
    实验
      {ds}
```"""


def _poster_svg(
    *,
    title: str,
    model_name: str,
    tldr_lines: list[str],
    key_lines: list[str],
) -> str:
    safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe_model = (model_name or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    tldr = tldr_lines[:3] if tldr_lines else []
    keys = key_lines[:3] if key_lines else []

    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    y = 90
    tldr_svg = []
    for i, line in enumerate(tldr):
        tldr_svg.append(f'<text x="60" y="{y + i*26}" class="b">• {esc(line)}</text>')

    ky = 430
    key_svg = []
    for i, line in enumerate(keys):
        key_svg.append(f'<text x="520" y="{ky + i*24}" class="s">• {esc(line)}</text>')

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540">
  <defs>
    <style>
      .t {{ font: 700 22px system-ui, -apple-system, Segoe UI, Arial; fill: #0f172a; }}
      .m {{ font: 600 14px system-ui, -apple-system, Segoe UI, Arial; fill: #334155; }}
      .b {{ font: 500 16px system-ui, -apple-system, Segoe UI, Arial; fill: #0f172a; }}
      .s {{ font: 500 14px system-ui, -apple-system, Segoe UI, Arial; fill: #0f172a; }}
      .cap {{ font: 600 12px system-ui, -apple-system, Segoe UI, Arial; fill: #475569; letter-spacing: .2px; }}
      .box {{ fill: #f8fafc; stroke: #e2e8f0; stroke-width: 1; rx: 10; }}
      .node {{ fill: #ffffff; stroke: #cbd5e1; stroke-width: 1; rx: 10; }}
      .arrow {{ stroke: #64748b; stroke-width: 2; fill: none; marker-end: url(#arrow); }}
    </style>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#64748b" />
    </marker>
  </defs>

  <rect x="20" y="20" width="920" height="500" class="box"/>
  <text x="50" y="62" class="t">{safe_title}</text>
  <text x="50" y="82" class="m">{safe_model}</text>

  <text x="50" y="115" class="cap">TL;DR</text>
  {''.join(tldr_svg)}

  <text x="50" y="210" class="cap">Pipeline</text>
  <rect x="60" y="235" width="150" height="52" class="node"/><text x="75" y="266" class="s">Text</text>
  <rect x="240" y="235" width="180" height="52" class="node"/><text x="255" y="266" class="s">SDP + Graphs</text>
  <rect x="450" y="235" width="200" height="52" class="node"/><text x="465" y="266" class="s">MV Encoder</text>
  <rect x="680" y="235" width="200" height="52" class="node"/><text x="695" y="266" class="s">Tree Decoder → Eq</text>
  <path d="M210 261 L240 261" class="arrow"/>
  <path d="M420 261 L450 261" class="arrow"/>
  <path d="M650 261 L680 261" class="arrow"/>

  <text x="510" y="410" class="cap">Key Notes</text>
  {''.join(key_svg)}
  <text x="50" y="500" class="cap">Generated by Paper2Post (local, heuristic extract)</text>
</svg>
"""


def _write_post(
    *,
    repo_root: Path,
    date_str: str,
    lang: str,
    categories: list[str],
    tags: list[str],
    slug: str,
    info: PaperInfo,
    poster_rel: str | None,
    include_mindmap: bool,
) -> Path:
    post_dir = repo_root / "_posts"
    post_dir.mkdir(parents=True, exist_ok=True)

    dt = datetime.fromisoformat(date_str)
    date_front = dt.strftime("%Y-%m-%d 00:00:00 +0800")
    title = info.title.strip()

    cat = ", ".join(categories) if categories else "PaperNotes"
    tag = ", ".join(tags) if tags else "paper"

    poster_md = f"![poster]({poster_rel})\n" if poster_rel else ""

    contributions_md = ""
    if info.contributions:
        contributions_md = "\n".join([f"- {c}" for c in info.contributions])
    else:
        contributions_md = "- （论文中未显式列出 contribution 列表；此处基于摘要/结论整理）"

    datasets_md = ", ".join(info.datasets) if info.datasets else "（未识别）"

    ref_lines = []
    if info.venue or info.year:
        ref_lines.append(f"- Venue: {info.venue} ({info.year})".strip())
    if info.doi:
        ref_lines.append(f"- DOI: {info.doi}")
    if info.keywords:
        ref_lines.append(f"- Keywords: {', '.join(info.keywords)}")
    ref_block = "\n".join(ref_lines) if ref_lines else "- Metadata: （PDF 未提供可用条目）"

    model = info.model_name or "MVG-DS-T"

    tldr = [
        f"提出多视图图学习到树的 MWP 求解模型 `{model}`，同时建模语义与数量关系两种图视图。",
        "用“结构重构”获得更适配下游任务的表示，用“跨视图对齐”提升多视图一致性。",
        "在 Math23K 与 MAWPS 上达到与 SOTA Graph2Tree 类方法相当的性能。"
        if info.datasets
        else "在常用 MWP 数据集上报告了与 SOTA 方法相当的结果。",
    ]

    body_lang_note = "" if lang.lower().startswith("zh") else "NOTE: This post is generated in Chinese by default.\n"

    mindmap = _mermaid_mindmap(model, info.datasets) if include_mindmap else ""

    md = f"""---
layout: post
title: "{title}"
date: {date_front}
categories: [{cat}]
tags: [{tag}]
toc: true
math: true
---

{body_lang_note}{poster_md}
## TL;DR
{chr(10).join([f'- {x}' for x in tldr])}

## 论文信息
{ref_block}

## 问题与动机
- 任务：数学应用题（Math Word Problem, MWP）从文本生成可执行的方程/表达式。
- 现有图方法常依赖先验知识图/手工构图：构图噪声与偏差会影响鲁棒性；多图视图学习容易“各学各的”，缺少统一表示。

## 方法概览（{model}）
{_mermaid_flow(model)}

### 图视图构建
- 语义图：基于语义依存分析（semantic dependency parsing）构建词/实体关系。
- 数量图：强调数量、单位、比较等与数值推理相关的边与结构。

### 双策略多视图学习
- 重构（Reconstruction）：对基准图结构进行重构学习，提取更适配下游表达式生成的表示。
- 对齐（Alignment）：跨视图约束语义/数量嵌入的一致性，缓解独立视图表示不统一的问题。

### 解码与训练细节
- 解码：树结构解码器（Graph2Tree 风格）自顶向下生成表达式。
- 损失：引入自适应长度归一化的平衡项，缓解不同表达式长度带来的训练偏置。

## 主要贡献（论文原文归纳）
{contributions_md}

## 实验与结果（高层总结）
- 数据集：{datasets_md}
- 结论：作者报告 `{model}` 的整体效果与当前主流 Graph2Tree 类图方法“可比/相当”，并通过消融/对比展示各模块有效性（详见论文实验章节）。

## 局限与展望（来自结论与个人解读）
- 仍依赖解析/构图质量：语义依存与数量抽取的误差会向下传递。
- 多视图策略的收益可能与数据集/题型分布相关；在更复杂的多步推理题上仍需验证。
- 未来方向：结合预训练模型、更强的图结构学习与外部知识整合（论文结论亦提及）。

{mindmap}

## 摘要（原文提取）
{info.abstract if info.abstract else '（未提取到摘要）'}

## 参考
- PDF 元数据来源：本地 PDF（Elsevier / ScienceDirect 导出）
"""

    out = post_dir / f"{dt.strftime('%Y-%m-%d')}-{slug}.md"
    out.write_text(md, encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a Chirpy post from a paper PDF (local heuristic).")
    parser.add_argument("--pdf", required=True, help="Path to PDF")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="Post date YYYY-MM-DD")
    parser.add_argument("--lang", default="zh-CN", help="Output language (default zh-CN)")
    parser.add_argument("--categories", default="PaperNotes", help="Comma-separated categories")
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    parser.add_argument("--slug", default="", help="Override slug")
    parser.add_argument("--no-poster", action="store_true", help="Disable poster.svg generation")
    parser.add_argument("--no-mindmap", action="store_true", help="Disable mermaid mindmap")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    repo_root = Path(__file__).resolve().parents[1]

    text, meta = _read_pdf_text(pdf_path)
    info = _extract_info(text, meta)
    slug = args.slug.strip() or _slugify(info.model_name or info.title)

    cache_dir = repo_root / "papers" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{slug}.json"
    cache = {
        "source_pdf": str(pdf_path),
        "text_sha256": _sha256(text),
        "extracted": {
            "title": info.title,
            "authors": info.authors,
            "venue": info.venue,
            "year": info.year,
            "doi": info.doi,
            "keywords": info.keywords,
            "model_name": info.model_name,
            "datasets": info.datasets,
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    poster_rel = None
    if not args.no_poster:
        assets_dir = repo_root / "assets" / "images" / "papers" / slug
        assets_dir.mkdir(parents=True, exist_ok=True)
        poster_path = assets_dir / "poster.svg"

        key_lines = []
        if info.contributions:
            key_lines = info.contributions[:3]
        else:
            if info.doi:
                key_lines.append(f"DOI: {info.doi}")
            if info.datasets:
                key_lines.append(f"Datasets: {', '.join(info.datasets)}")
            key_lines.append("Dual strategies: reconstruction + alignment")

        poster_svg = _poster_svg(
            title=info.title,
            model_name=info.model_name or "",
            tldr_lines=[
                f"{info.model_name or '模型'}：多视图图学习到树的 MWP 求解器",
                "两类基准图：语义依存图 + 数量图",
                "双策略：结构重构（自适应）+ 跨视图对齐（一致性）",
            ],
            key_lines=key_lines,
        )
        poster_path.write_text(poster_svg, encoding="utf-8")
        poster_rel = f"/assets/images/papers/{slug}/poster.svg"

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    if not tags:
        if info.model_name:
            tags.append(info.model_name.lower())
        tags.extend(["math-word-problem", "graph-learning"])

    post_path = _write_post(
        repo_root=repo_root,
        date_str=args.date,
        lang=args.lang,
        categories=categories,
        tags=tags,
        slug=slug,
        info=info,
        poster_rel=poster_rel,
        include_mindmap=not args.no_mindmap,
    )

    print(str(post_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

