# 博客项目 Skills

本目录包含博客专用的 agent skills，专注于**论文解析→博客写作**工作流。

## 论文转博客工作流

### 完整流程

```
论文 (arXiv URL/PDF) 
    ↓
【1. 下载解析】/read-arxiv-paper 或 /pdf
    ↓
【2. 图片提取】/extract-paper-images
    ↓
【3. 内容总结】/summarize
    ↓
【4. 博客写作】/blog-writer
    ↓
【5. SEO 优化】/seo-content-writer
    ↓
【6. 社交推广】/social-content-generator
```

---

## Skills 列表

### 📄 论文处理核心 Skills

| Skill | 功能 | 用途 |
|-------|------|------|
| `/pdf` | PDF 解析 | 提取文本、表格、图片 |
| `/summarize` | 内容摘要 | 长文档总结、URL 摘要 |
| `/read-arxiv-paper` | arXiv 下载 | 下载 arXiv 论文源码 |
| `/extract-paper-images` | 图片提取 | 从论文中提取图表 |

### ✍️ 博客写作 Skills

| Skill | 功能 | 用途 |
|-------|------|------|
| `/blog-writer-0.1.0` | 博客写作 | 将技术内容转换为博客文章 |
| `/seo-content-writer-2.0.0` | SEO 优化 | 优化文章 SEO |
| `/content-strategy-0.1.0` | 内容策略 | 规划内容结构 |
| `/copywriting-0.1.0` | 文案写作 | 撰写吸引人的标题 |
| `/writing-plans-0.1.0` | 写作计划 | 制定写作大纲 |
| `/research-paper-writer-0.1.0` | 论文写作 | 学术论文格式 |

### 📢 社交推广 Skills

| Skill | 功能 | 用途 |
|-------|------|------|
| `/social-content-generator-0.1.0` | 社交内容 | 生成 Twitter/LinkedIn 文案 |
| `/social-media-scheduler-1.0.0` | 发布调度 | 安排社交媒体发布计划 |

### 💡 创意策划 Skills

| Skill | 功能 | 用途 |
|-------|------|------|
| `/brainstorming-0.1.0` | 头脑风暴 | 创意发散和需求澄清 |
| `/seo-1.0.3` | SEO 审计 | 站点 SEO 分析 |

---

## 使用示例

### 示例 1：处理 arXiv 论文

```bash
# 用户：帮我处理这篇论文 https://arxiv.org/abs/2301.07372

# 步骤 1：下载并解析论文
/read-arxiv-paper https://arxiv.org/abs/2301.07372

# 步骤 2：提取所有图片
/extract-paper-images 2301.07372

# 步骤 3：生成摘要
/summarize ~/.cache/arxiv/2301.07372/main.tex --length long

# 步骤 4：撰写博客
/blog-writer 根据上述论文摘要，写一篇面向开发者的技术博客

# 步骤 5：SEO 优化
/seo-content-writer 优化这篇文章，关键词包括 "transformer", "attention mechanism"
```

### 示例 2：处理本地 PDF

```bash
# 用户：帮我总结这个 PDF 文件 E:\papers\attention.pdf

# 步骤 1：提取文本和图片
/pdf extract_text E:\papers\attention.pdf
/pdf extract_images E:\papers\attention.pdf

# 步骤 2：总结
/summarize E:\papers\attention.txt --length long

# 步骤 3：博客写作
/blog-writer 根据上述内容写一篇博客
```

---

## 注意事项

1. **PDF 依赖**：`/pdf` skill 需要安装 `pypdf`、`pdfplumber` 等 Python 库
2. **arXiv 缓存**：下载的论文缓存在 `~/.cache/arxiv/`，避免重复下载
3. **图片格式**：提取的图片保存到 `images/` 目录，自动生成索引文件
4. **摘要长度**：使用 `--length short|medium|long|xl|xxl` 控制摘要长度

---

## 已移除的 Skills

以下 skills 因需要外部 Token 服务已移除：
- ❌ `autoglm-*` 系列（需要 AutoGLM Token 服务）
- ❌ `aminer-open-academic`（需要 AMiner API Token）

如需类似功能，可使用：
- 网络搜索：手动搜索后使用 `/summarize` 总结
- 学术数据：使用 `/read-arxiv-paper` 直接获取论文
