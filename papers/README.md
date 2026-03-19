# Paper2Post inbox

把论文题目/链接（可选附 PDF）放进 `papers/inbox/`，自动化会生成一篇可发布到本仓库（Jekyll/Chirpy）的博客文章并提交到 git 分支。

## 输入格式

支持两种请求文件（建议 YAML）：

### 1) YAML（推荐）

放一个 `*.yml` / `*.yaml` 到 `papers/inbox/`，字段示例见 `papers/examples/request.yml`。

常用字段：

- `title`: 论文标题（可选）
- `url`: 论文链接（可选，arXiv/DOI/Publisher 都行）
- `pdf`: PDF 相对路径（可选；例如 `papers/inbox/foo.pdf`）
- `lang`: 输出语言（默认 `zh-CN`）
- `categories`: Chirpy categories（数组，可选）
- `tags`: Chirpy tags（数组，可选）
- `out_date`: 文章日期（`YYYY-MM-DD`，可选；默认当天）
- `include_poster`: 是否生成 `poster.svg`（默认 `true`）
- `include_mindmap`: 是否生成 Mermaid mindmap（默认 `true`）

### 2) TXT（最简）

放一个 `*.txt` 到 `papers/inbox/`，第一行写论文题目或链接即可。

## PDF 放置规则（可选）

自动化会按优先级查找 PDF：

1. 请求文件里显式指定的 `pdf`
2. 与请求文件同名的 `*.pdf`（例如 `foo.yml` 对应 `foo.pdf`）

## 输出产物（约定）

- 文章：`_posts/YYYY-MM-DD-<slug>.md`
- 配图：`assets/images/papers/<slug>/poster.svg`（可选）
- 请求归档：`papers/done/`
- 缓存（增量）：`papers/cache/`（用于避免重复处理）

