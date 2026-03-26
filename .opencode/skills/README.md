# ⚠️ Skills 目录保护说明

**生效日期：** 2026-03-26  
**保护状态：** ✅ 已启用

---

## 🛡️ 保护配置

### OpenCode 配置

配置文件：`.opencode/config.json`

```json
{
  "skills": {
    "readonly": true,
    "protected_paths": [".opencode/skills/"],
    "require_permission_for": ["write", "delete", "move", "mkdir"],
    "message": "⚠️ Skills 目录受保护，修改需要先询问用户权限"
  }
}
```

### 保护规则

**禁止的操作（需授权）：**
- ❌ 修改任何 `.md` 文件
- ❌ 创建新文件/目录
- ❌ 删除文件/目录
- ❌ 移动文件或重命名

**允许的操作：**
- ✅ 读取 skill 文件
- ✅ 执行 skill 命令
- ✅ 生成内容到 `_posts/` 目录

---

## 📦 已安装的 Skills (16 个)

### 博客写作核心

| Skill | 用途 | 状态 |
|-------|------|------|
| `blog-writer-0.1.0` | 博客写作 | ✅ 已恢复原始版本 |
| `seo-content-writer-2.0.0` | SEO 优化 | ✅ |
| `content-strategy-0.1.0` | 内容策略 | ✅ |
| `copywriting-0.1.0` | 文案写作 | ✅ |
| `writing-plans-0.1.0` | 写作计划 | ✅ |
| `social-content-generator-0.1.0` | 社交内容 | ✅ |
| `social-media-scheduler-1.0.0` | 社交调度 | ✅ |
| `brainstorming-0.1.0` | 头脑风暴 | ✅ |
| `seo-1.0.3` | SEO 审计 | ✅ |
| `research-paper-writer-0.1.0` | 论文写作 | ✅ |

### 论文处理

| Skill | 用途 | 状态 |
|-------|------|------|
| `paper-analyzer` | 论文分析（新增） | ✅ 已创建 |
| `paper-to-blog` | 论文博客转换（新增） | ✅ 已创建 |
| `extract-paper-images` | 论文图片提取 | ✅ |
| `pdf` | PDF 解析 | ✅ |
| `read-arxiv-paper` | arXiv 论文下载 | ✅ |
| `summarize` | 内容摘要 | ✅ |

---

## 🆕 新增技能说明

### paper-analyzer（2026-03-26 创建）

**职责**: 通用论文分析，从多种来源提取信息

**支持来源**:
- arXiv 链接
- GitHub 仓库
- PDF 文件
- 项目主页
- 论文标题（搜索）

**输出**: `analysis.json` 结构化结果

### paper-to-blog（2026-03-26 创建）

**职责**: 将分析结果转换为 Chirpy 博客格式

**特点**:
- 硬编码 Chirpy 规范（Frontmatter、图片说明）
- 可配置写作风格（`config/style-guide.md`）
- 自动发布前检查

**已迁移自 blog-writer**:
- ✅ 写作风格指南
- ✅ 段落组织建议
- ✅ 词汇选择偏好

### blog-writer 状态

**建议**: 保留用于非论文类博客写作

**分工**:
- `paper-to-blog`: 论文阅读笔记
- `blog-writer`: 原创文章、思考、教程

---

## 📝 blog-writer-0.1.0 恢复说明

### 恢复原因
- 之前修改了目录结构（创建 `references/`）
- 违反了 skills 保护原则
- 需要恢复原始版本

### 恢复操作
1. ✅ 删除修改后的版本
2. ✅ 从 Git 提交 `72c7bed` 恢复
3. ✅ 恢复原始文件结构

### 当前状态
```
blog-writer-0.1.0/
├── SKILL.md                      # 原始版本
├── README.md                     # 原始版本
├── _meta.json                    # 原始版本
├── manage_examples.py            # 原始版本
├── style-guide.md                # 原始版本
├── 2024-02-17-radical-transparency-sales.md
├── 2024-02-17-raycast-spotlight-superpowers.md
├── 2024-02-17-short-form-content-marketing.md
├── 2024-02-17-typing-speed-benefits.md
├── 2024-03-14-effective-ai-prompts.md
├── 2024-11-08-ai-revolutionizing-entry-level-sales.md
└── 2025-11-12-why-ai-art-is-useless.md
```

**注意：** 示例文章在根目录，**没有** `references/` 子目录

---

## 🔐 修改授权流程

当需要修改 skills 时，agent 必须：

1. **说明原因** - 为什么需要修改
2. **列出变更** - 具体修改哪些文件
3. **提供替代** - 是否有其他方案
4. **等待确认** - 用户明确同意后才执行

### 示例询问格式

```
⚠️ 检测到需要修改 Skills

文件：.opencode/skills/blog-writer-0.1.0/SKILL.md
原因：[说明为什么需要修改]
修改内容：[具体修改内容]
影响范围：[影响哪些功能]
替代方案：[其他方法]

是否继续？[是/否/查看详情]
```

---

## 📊 修改历史

| 日期 | Skill | 修改内容 | 授权状态 |
|------|-------|---------|---------|
| 2026-03-26 12:16 | blog-writer-0.1.0 | 创建 references/目录 | ❌ 未授权 |
| 2026-03-26 14:19 | blog-writer-0.1.0 | 从 Git 恢复原始版本 | ✅ 用户同意 |

---

## ⚠️ 注意事项

1. **不要手动修改 skills** - 通过 Git 或授权流程
2. **需要修改时先询问** - agent 会主动询问权限
3. **配置已保存** - `.opencode/config.json` 已启用保护

---

## 🔗 相关文档

- [config.json](../config.json) - OpenCode 配置
- [blog-writer-0.1.0/SKILL.md](./blog-writer-0.1.0/SKILL.md) - Blog Writer Skill 定义

---

*最后更新：2026-03-26*
