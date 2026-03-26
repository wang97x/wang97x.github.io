# paper-analyzer 快速开始指南

## 安装依赖

```bash
pip install pdfplumber PyMuPDF pyyaml
```

## 快速使用

### 一键分析（推荐）

```bash
python .opencode/skills/paper-analyzer/scripts/analyze_paper.py \
  "https://arxiv.org/abs/2305.01879" \
  --categories "论文阅读，知识蒸馏" \
  --next paper-to-blog
```

### 分步执行

#### 1. 下载论文

```bash
python .opencode/skills/paper-analyzer/scripts/download_paper.py \
  "https://arxiv.org/abs/2305.01879" \
  "_papers/scott-kd"
```

#### 2. 解析内容

```bash
python .opencode/skills/paper-analyzer/scripts/parse_paper.py \
  "_papers/scott-kd"
```

输出：`_papers/scott-kd/analysis.json`

#### 3. 提取图片

```bash
python .opencode/skills/paper-analyzer/scripts/extract_figures.py \
  "_papers/scott-kd/paper.pdf" \
  "_papers/scott-kd/figures" \
  --analysis "_papers/scott-kd/analysis.json"
```

## 支持的输入

| 类型 | 示例 | 说明 |
|------|------|------|
| **arXiv** | `https://arxiv.org/abs/2305.01879` | ✅ 完整支持 |
| **GitHub** | `https://github.com/user/repo` | ⚠️ 基础支持 |
| **PDF** | `paper.pdf` 或 URL | ✅ 完整支持 |
| **网页** | `https://example.com/paper` | ⚠️ 有限支持 |

## 输出结构

```
_papers/{slug}/
├── analysis.json          # 核心输出
├── paper.pdf              # 下载的 PDF
└── figures/
    ├── fig_1_1.png
    ├── fig_2_1.png
    └── ...
```

## analysis.json 结构

```json
{
  "metadata": {
    "title": "...",
    "authors": ["..."],
    "affiliations": ["..."],
    "venue": "...",
    "date": "..."
  },
  "abstract": "...",
  "sections": {
    "introduction": "...",
    "method": "...",
    "experiments": "..."
  },
  "figures": [
    {
      "path": "...",
      "suggested_name": "roadmap.png",
      "type": "overview",
      "caption": "..."
    }
  ],
  "key_contributions": ["...", "..."],
  "tags": ["LLM", "..."]
}
```

## 图片分类

自动将图片分为 6 类：

| 类型 | 关键词 | 典型页面 |
|------|--------|---------|
| **overview** | roadmap, overview, pipeline | 1-2 |
| **architecture** | framework, architecture, model | 2-4 |
| **method** | method, approach, algorithm | 3-6 |
| **result** | result, comparison, accuracy | 7-12 |
| **ablation** | ablation, study, analysis | 8-12 |
| **table** | table, statistic, benchmark | 任意 |

## 常见问题

### Q: 下载失败
**A**: 检查网络连接，arXiv 可能需要代理

### Q: 解析结果为空
**A**: 可能是扫描版 PDF，需要 OCR 支持（暂未实现）

### Q: 图片提取失败
**A**: 检查 PyMuPDF 版本，确保 `pip install PyMuPDF`

### Q: JSON 格式错误
**A**: 检查 Python 版本（需要 3.8+）

## 下一步

分析完成后，使用 paper-to-blog 转换为博客：

```bash
python .opencode/skills/paper-to-blog/scripts/convert_to_blog.py \
  "_papers/scott-kd/analysis.json" \
  --categories "论文阅读，知识蒸馏" \
  --output "_posts/"
```

验证生成的博客：

```bash
python .opencode/skills/paper-to-blog/scripts/validate_blog.py \
  "_posts/2024-03-26-scott-kd.md"
```
