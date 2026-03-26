---
name: paper-analyzer
description: 通用论文分析技能，从多种来源（arXiv/GitHub/PDF/网页）提取论文信息并生成结构化分析结果。**优先使用于**: (1) 用户需要**分析/总结/理解论文内容**，(2) 用户提供 arXiv/GitHub/PDF 链接并提到"论文阅读/分析/总结"，(3) 需要将论文转换为博客文章，(4) 需要提取论文的元数据/章节/图片。**注意**: 如果用户只是需要合并/分割/旋转 PDF 文件操作，使用 pdf skill；如果用户需要**理解论文内容**，使用 paper-analyzer
---

# Paper Analyzer - 通用论文分析技能

## 核心职责

从多种来源提取论文信息，输出结构化 JSON 分析结果供 paper-to-blog 使用。

## 支持的输入来源

1. **arXiv 链接**: `https://arxiv.org/abs/2305.01879` → 下载 PDF + LaTeX 源码
2. **GitHub 仓库**: `https://github.com/user/repo` → 提取 README + 下载论文 PDF
3. **项目主页**: `https://example.com/paper` → 抓取页面内容 + 查找 PDF 链接
4. **PDF 文件**: 本地路径或 URL → 直接解析
5. **论文标题 + 作者**: 尝试网络搜索后让用户确认

## 工作流程

### 1. 来源检测与下载

```bash
# 调用下载脚本
python ".opencode/skills/paper-analyzer/scripts/download_paper.py" \
  "<input_source>" \
  "_papers/<paper-slug>/"
```

**下载策略**（带 fallback）:
- arXiv: PDF → e-print 源码 → GitHub
- GitHub: README → releases → 项目主页 → 搜索
- 网页：HTML → 查找 PDF 链接 → 下载
- 搜索：Semantic Scholar → Google Scholar → arXiv

**验证要求**:
- 文件大小 > 1KB
- PDF 文件头正确（`%PDF-` 开头）
- 下载失败自动重试（3 次，间隔递增）

### 2. 内容解析

```bash
# 调用解析脚本
python ".opencode/skills/paper-analyzer/scripts/parse_paper.py" \
  "_papers/<paper-slug>/"
```

**解析器 fallback 链**:
1. pdfplumber（首选，支持文本提取）
2. PyMuPDF/fitz（备选，支持扫描版）
3. pdfminer（第三选择）
4. OCR/tesseract（最后手段，针对图片型 PDF）

**提取内容**:
- 元数据（标题、作者、机构、日期）
- 摘要（Abstract）
- 各章节内容（Introduction, Method, Experiments, Conclusion）
- 参考文献列表

### 3. 图片提取与分类

```bash
# 调用图片提取脚本
python ".opencode/skills/paper-analyzer/scripts/extract_figures.py" \
  "_papers/<paper-slug>/paper.pdf" \
  "_papers/<paper-slug>/figures/"
```

**图片分类规则**:
| 类型 | 识别关键词 | 示例 |
|------|-----------|------|
| overview | roadmap, overview, pipeline | 论文路线图 |
| architecture | framework, architecture, model | 架构图 |
| method | method, approach, algorithm | 方法示意图 |
| result | results, comparison, accuracy | 实验结果 |
| ablation | ablation, study, analysis | 消融实验 |
| table | table, statistics, benchmark | 数据表格 |

**输出格式**:
```json
{
  "path": "_papers/slug/figures/fig_1_1.png",
  "suggested_name": "roadmap.png",
  "caption": "论文内容路线图",
  "type": "overview",
  "page": 1,
  "confidence": "high"
}
```

### 4. 生成分析 JSON

**输出文件**: `_papers/<paper-slug>/analysis.json`

**完整结构**:
```json
{
  "metadata": {
    "title": "论文英文标题",
    "authors": ["Author One", "Author Two"],
    "affiliations": ["Institution 1", "Institution 2"],
    "venue": "会议/期刊名称",
    "date": "2024-01-15",
    "links": {
      "arxiv": "https://arxiv.org/abs/...",
      "github": "https://github.com/...",
      "homepage": "https://..."
    }
  },
  "abstract": "论文摘要原文",
  "sections": {
    "introduction": "引言内容",
    "background": "背景知识",
    "method": "核心方法",
    "experiments": "实验设置与结果",
    "conclusion": "结论与未来工作"
  },
  "figures": [
    {"path": "...", "suggested_name": "...", "caption": "...", "type": "..."}
  ],
  "tables": [...],
  "key_contributions": [
    "贡献 1",
    "贡献 2",
    "贡献 3"
  ],
  "tags": ["LLM", "Knowledge Distillation", "Reasoning"],
  "paper_slug": "paper-slug-for-url"
}
```

## 输出位置

```
_papers/<paper-slug>/
├── analysis.json          # 结构化分析结果（核心输出）
├── paper.pdf              # 下载的 PDF 文件
└── figures/               # 提取的图片
    ├── fig_1_1.png
    ├── fig_2_1.png
    └── ...
```

## 与 paper-to-blog 的协作

**标准调用流程**:
1. 用户提供 arXiv 链接
2. 调用 paper-analyzer 生成 `analysis.json`
3. 自动调用 paper-to-blog 读取 JSON 并生成博客
4. 询问用户是否保留临时文件

**示例命令**:
```bash
# 完整流程
python ".opencode/skills/paper-analyzer/scripts/analyze_paper.py" \
  "https://arxiv.org/abs/2305.01879" \
  --output "_papers/scott-kd" \
  --next "paper-to-blog" \
  --categories "论文阅读，知识蒸馏"
```

## 错误处理策略

| 错误级别 | 示例 | 处理方式 |
|---------|------|---------|
| **致命错误** | 无法下载 PDF、无可用内容 | 终止，告知用户原因 |
| **严重错误** | 所有解析器失败、图片提取全失败 | 降级处理，询问继续/终止 |
| **警告** | 某些章节缺失、图片命名置信度低 | 继续，在报告中说明 |
| **提示** | 临时文件未清理 | 自动处理或询问 |

## 临时文件管理

**默认策略**: 发布后询问用户是否删除 `_papers/<slug>/` 目录

**自动清理条件**:
- 用户配置 `auto_cleanup: true`
- 博客生成成功且用户确认无误

**.gitignore 规则**:
```
_papers/*.json
_papers/*.py
_papers/figures/
*.tmp
```

## 参考文档

- **详细 API**: 详见 `references/api.md`
- **解析器配置**: 详见 `references/parsers.md`
- **图片分类规则**: 详见 `references/figure-classification.md`

## 示例

### 输入
```
https://arxiv.org/abs/2305.01879 分类：论文阅读，知识蒸馏
```

### 输出
```
_papers/scott-kd/
├── analysis.json (8.2KB)
├── paper.pdf (1.2MB)
└── figures/
    ├── fig_1_1.png (roadmap.png)
    ├── fig_3_1.png (contrastive_decoding.png)
    └── ...
```

### 分析 JSON 片段
```json
{
  "metadata": {
    "title": "Self-Consistent Thought Distillation",
    "authors": ["Xiaohan Xu", "Ming Li"],
    "affiliations": ["University A", "University B"],
    "venue": "ICML 2024",
    "date": "2024-05-15",
    "links": {
      "arxiv": "https://arxiv.org/abs/2305.01879",
      "github": "https://github.com/user/scott"
    }
  },
  "key_contributions": [
    "提出自洽思维链蒸馏方法",
    "在 5 个基准上达到 SOTA",
    "开源代码和数据"
  ],
  "tags": ["LLM", "Knowledge Distillation", "Chain-of-Thought", "Self-Consistency"]
}
```
