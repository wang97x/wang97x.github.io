---
name: read-arxiv-paper
description: 下载并解析 arXiv 论文。从 arXiv URL 获取 LaTeX 源码，提取文本内容用于后续总结和分析。
---

# Read ArXiv Paper

## 功能
从 arXiv 下载论文源码并提取文本内容。

## 使用方法

### 输入格式
支持以下格式的 arXiv URL：
- `https://arxiv.org/abs/2301.07372`
- `https://arxiv.org/pdf/2301.07372.pdf`
- `arXiv:2301.07372`
- `2301.07372`

### 工作流程

1. **解析 arXiv ID** - 从 URL 提取论文 ID
2. **下载源码** - 从 `https://arxiv.org/e-print/{arxiv_id}` 下载 `.tar.gz`
3. **解压文件** - 解压到临时目录
4. **查找入口** - 定位 `main.tex` 或其他 `.tex` 文件
5. **提取文本** - 读取 LaTeX 源码内容

### 输出
- 论文元数据（标题、作者、摘要）
- LaTeX 源码文件路径
- 提取的文本内容

## 示例

```bash
# 下载并解析论文
/read-arxiv-paper https://arxiv.org/abs/2301.07372

# 输出示例：
# 论文标题：Attention Is All You Need
# 作者：Vaswani et al.
# 源码目录：~/.cache/arxiv/2301.07372/
# 主要文件：main.tex
```

## 后续处理

下载完成后，可以：
1. 使用 `/summarize` 总结论文内容
2. 使用 `/extract-paper-images` 提取图片
3. 使用 `/pdf` 处理 PDF 版本

## 缓存
下载的论文会缓存在 `~/.cache/arxiv/{arxiv_id}/`，避免重复下载。
