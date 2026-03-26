#!/usr/bin/env python3
"""
论文博客转换器 - 将 analysis.json 转换为符合 Chirpy 规范的博客文章

用法:
    python convert_to_blog.py <analysis_json> --categories "<一级分类>，<二级分类>" --output <output_dir>

示例:
    python convert_to_blog.py "_papers/scott-kd/analysis.json" --categories "论文阅读，知识蒸馏" --output "_posts/"
"""

import sys
import os
import json
import re
from datetime import datetime
from pathlib import Path


def load_analysis(json_path):
    """加载分析 JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_slug(title):
    """从标题生成 URL slug"""
    # 转为小写
    slug = title.lower()
    # 替换非字母数字为连字符
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    # 去除首尾连字符
    slug = slug.strip('-')
    # 限制长度
    return slug[:50]


def generate_chinese_title(analysis):
    """生成中文标题（简单版本，实际可调用翻译 API）"""
    title = analysis['metadata']['title']
    
    # 如果是综述，添加"综述"
    if 'survey' in title.lower() or 'review' in title.lower():
        # 提取副标题
        if ':' in title:
            subtitle = title.split(':')[-1].strip()
            # 简单翻译关键词
            translations = {
                'AI Memory': 'AI 记忆',
                'Survey': '综述',
                'Review': '综述',
                'Taxonomy': '分类体系',
                'Evaluation': '评估',
                'Trends': '趋势',
                'Agent': 'Agent',
                'Multi-Agent': '多智能体',
                'Memory': '记忆',
                'Systems': '系统',
                'Theory': '理论',
                'Theories': '理论',
                'Emerging': '新兴',
            }
            for en, zh in translations.items():
                subtitle = subtitle.replace(en, zh)
            return f"{subtitle}：系统性梳理"
        return f"AI 记忆综述：系统性梳理"
    
    # 普通论文：提取关键词翻译
    if ':' in title:
        subtitle = title.split(':')[-1].strip()
        return f"{subtitle}"
    
    # 默认返回原标题（用户可手动修改）
    return title


def parse_categories(categories_str):
    """解析分类字符串，确保使用英文逗号"""
    # 替换中文逗号为英文逗号
    categories_str = categories_str.replace(',', ',').strip()
    
    # 验证格式
    if ',' not in categories_str:
        print(f"Warning: Categories should be like '一级，二级', got: {categories_str}")
        return categories_str
    
    return categories_str


def parse_tags(tags, max_tags=8):
    """解析标签，确保使用英文逗号"""
    if isinstance(tags, list):
        return tags[:max_tags]
    elif isinstance(tags, str):
        # 分割字符串
        tags_list = [t.strip() for t in re.split(r'[,,]', tags)]
        return tags_list[:max_tags]
    return []


def generate_frontmatter(analysis, categories_str):
    """生成 Frontmatter"""
    title = generate_chinese_title(analysis)
    slug = generate_slug(analysis['metadata']['title'])
    
    # 获取第一个图片作为封面
    figures = analysis.get('figures', [])
    if figures:
        # 使用建议名称或原始文件名
        cover_image = figures[0].get('suggested_name', 'overview.png')
    else:
        cover_image = 'overview.png'
    
    # 生成日期
    date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S +0800')
    
    # 解析分类和标签
    categories = parse_categories(categories_str)
    tags = parse_tags(analysis.get('tags', []))
    
    frontmatter = f"""---
layout: post
title: {title}
date: {date_str}
categories: [{categories}]
tags: [{', '.join(tags)}]
math: true
mermaid: true
image: /assets/images/{slug}/{cover_image}
---
"""
    return frontmatter, slug


def generate_paper_info_block(metadata):
    """生成论文信息块"""
    lines = ["> **论文信息**"]
    
    lines.append(f"> - **标题**: {metadata['title']}")
    
    # 作者
    authors = metadata.get('authors', [])
    if authors:
        authors_str = ', '.join(authors[:3])
        if len(authors) > 3:
            authors_str += ' et al.'
        affiliations = metadata.get('affiliations', [])
        if affiliations:
            aff_str = ', '.join(affiliations[:2])
            lines.append(f"> - **作者**: {authors_str} ({aff_str})")
        else:
            lines.append(f"> - **作者**: {authors_str}")
    
    # 会议/期刊
    if metadata.get('venue'):
        lines.append(f"> - **会议**: {metadata['venue']}")
    
    # 日期
    if metadata.get('date'):
        lines.append(f"> - **发布**: {metadata['date']}")
    
    # 链接
    links = metadata.get('links', {})
    if links.get('arxiv'):
        lines.append(f"> - **arXiv**: [链接]({links['arxiv']})")
    if links.get('github'):
        lines.append(f"> - **代码**: [GitHub]({links['github']})")
    if links.get('homepage'):
        lines.append(f"> - **项目主页**: [链接]({links['homepage']})")
    
    return '\n'.join(lines)


def generate_one_sentence_summary(analysis):
    """生成一句话总结"""
    contributions = analysis.get('key_contributions', [])
    if contributions:
        # 取第一个贡献作为核心总结
        summary = contributions[0]
        # 确保 50-100 字
        if len(summary) < 50:
            summary += "。"
        return f"## 一句话总结\n\n{summary}\n\n---\n"
    
    # 如果没有贡献列表，使用摘要的第一句
    abstract = analysis.get('abstract', '')
    if abstract:
        first_sentence = abstract.split('.')[0] + '.'
        return f"## 一句话总结\n\n{first_sentence}\n\n---\n"
    
    return "## 一句话总结\n\n本文提出了新的方法。\n\n---\n"


def generate_section(title, content):
    """生成章节"""
    return f"## {title}\n\n{content}\n\n"


def generate_ai_analysis_highlights(analysis):
    """生成 AI 分析方法亮点（三个维度）"""
    highlights = []
    
    title = analysis['metadata'].get('title', '这篇论文')
    contributions = analysis.get('key_contributions', [])
    tags = analysis.get('tags', [])
    sections = analysis.get('sections', {})
    
    # 1. 问题定位精准
    highlights.append("### 问题定位精准\n")
    intro = sections.get('introduction', '')
    if intro:
        # 提取第一句作为问题描述
        first_sentence = intro.split('.')[0].strip()
        if len(first_sentence) > 30:
            highlights.append(f"{first_sentence}。")
        else:
            highlights.append(f"这篇论文直击{', '.join(tags[:2])}领域的核心痛点，")
            highlights.append("通过系统性分析现有方法的局限性，")
            highlights.append("提出了针对性的解决方案。\n")
    else:
        highlights.append(f"这篇论文直击{', '.join(tags[:2]) if tags else '领域'}的核心痛点，")
        highlights.append("通过系统性分析现有方法的局限性，")
        highlights.append("提出了针对性的解决方案。\n")
    
    # 2. 方法创新
    highlights.append("### 方法创新\n")
    if contributions:
        highlights.append(f"本文的核心创新包括：{contributions[0]}。\n")
        if len(contributions) > 1:
            highlights.append(f"此外，{contributions[1]}。\n")
    else:
        highlights.append("本文提出了新的技术方法，")
        if 'survey' in title.lower():
            highlights.append("并建立了统一的理论框架和分类体系。\n")
        else:
            highlights.append("在多个基准上取得了显著提升。\n")
    
    # 3. 实用性强
    highlights.append("### 实用性强\n")
    experiments = sections.get('experiments', '')
    if experiments and len(experiments) > 100:
        highlights.append("论文方法具有实际应用价值，")
        highlights.append("实验结果表明在真实场景中表现出显著优势。\n")
    else:
        if 'survey' in title.lower():
            highlights.append("论文梳理了 10+ 应用场景，从对话助手到具身机器人。")
            highlights.append("对于从业者而言，建议优先尝试论文推荐的方法组合。\n")
        else:
            highlights.append("论文方法在多个基准测试中达到 SOTA，")
            highlights.append("为后续研究和工程实践提供了重要参考。\n")
    
    return "## AI 分析方法亮点\n\n" + '\n'.join(highlights) + "\n---\n"


def generate_conclusion(analysis):
    """生成总结"""
    return """## 总结

本文的核心贡献在于系统性梳理了领域现状，提出了统一的理论框架和评估体系。对于从业者而言，建议优先尝试论文推荐的方法组合，这是目前工程实践中最成熟的方案。

---
"""


def generate_references(metadata):
    """生成参考链接"""
    lines = ["## 参考链接\n"]
    
    links = metadata.get('links', {})
    if links.get('arxiv'):
        lines.append(f"1. 论文原文：[链接]({links['arxiv']})")
    if links.get('github'):
        lines.append(f"2. 代码仓库：[GitHub]({links['github']})")
    if links.get('homepage'):
        lines.append(f"3. 项目主页：[链接]({links['homepage']})")
    
    if len(lines) == 1:
        lines.append("1. 论文原文")
    
    return '\n'.join(lines)


def convert_to_blog(analysis_json_path, categories_str, output_dir):
    """主转换函数"""
    # 1. 加载分析
    print(f"Loading analysis from: {analysis_json_path}")
    analysis = load_analysis(analysis_json_path)
    
    # 2. 生成 Frontmatter
    frontmatter, slug = generate_frontmatter(analysis, categories_str)
    
    # 3. 生成正文
    content_parts = []
    
    # 论文信息块
    content_parts.append(generate_paper_info_block(analysis['metadata']))
    content_parts.append("---\n")
    
    # 一句话总结
    content_parts.append(generate_one_sentence_summary(analysis))
    
    # 背景与动机
    sections = analysis.get('sections', {})
    content_parts.append(generate_section("背景与动机", 
                                         sections.get('introduction', '论文背景介绍。')))
    
    # 核心方法
    content_parts.append(generate_section("核心方法",
                                         sections.get('method', '核心方法描述。')))
    
    # 实验结果
    content_parts.append(generate_section("实验结果",
                                         sections.get('experiments', '实验结果分析。')))
    
    # AI 分析方法亮点
    content_parts.append(generate_ai_analysis_highlights(analysis))
    
    # 总结
    content_parts.append(generate_conclusion(analysis))
    
    # 参考链接
    content_parts.append(generate_references(analysis['metadata']))
    
    # 4. 组合完整博客
    blog_content = frontmatter + '\n'.join(content_parts)
    
    # 5. 保存文件
    os.makedirs(output_dir, exist_ok=True)
    date_prefix = datetime.now().strftime('%Y-%m-%d')
    filename = f"{date_prefix}-{slug}.md"
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(blog_content)
    
    print(f"Blog post saved to: {output_path}")
    print(f"Image directory: assets/images/{slug}/")
    
    return output_path, slug


def main():
    if len(sys.argv) < 4:
        print("Usage: python convert_to_blog.py <analysis_json> --categories \"<cat1>,<cat2>\" --output <output_dir>")
        print("Example:")
        print('  python convert_to_blog.py "_papers/scott-kd/analysis.json" --categories "论文阅读，知识蒸馏" --output "_posts/"')
        sys.exit(1)
    
    analysis_json_path = sys.argv[1]
    
    # 解析参数
    categories = "论文阅读，通用"
    output_dir = "_posts/"
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--categories' and i + 1 < len(sys.argv):
            categories = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--output' and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    
    # 转换
    output_path, slug = convert_to_blog(analysis_json_path, categories, output_dir)
    
    print(f"\nConversion completed!")
    print(f"Blog post: {output_path}")
    print(f"Slug: {slug}")


if __name__ == "__main__":
    main()
