#!/usr/bin/env python3
"""
论文解析脚本 - 从 PDF 提取文本和元数据

用法:
    python parse_paper.py <paper_dir>

示例:
    python parse_paper.py "_papers/scott-kd/"
"""

import sys
import os
import json
import re
from pathlib import Path


def parse_pdf_with_pdfplumber(pdf_path):
    """使用 pdfplumber 解析 PDF"""
    try:
        import pdfplumber
        
        text_content = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    text_content.append({
                        'page': i + 1,
                        'text': text
                    })
        
        if text_content:
            return '\n\n'.join([p['text'] for p in text_content]), text_content
        return None, None
    except Exception as e:
        print(f"pdfplumber failed: {e}")
        return None, None


def parse_pdf_with_pymupdf(pdf_path):
    """使用 PyMuPDF (fitz) 解析 PDF"""
    try:
        import fitz
        
        text_content = []
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            text = page.get_text()
            if text:
                text_content.append({
                    'page': i + 1,
                    'text': text
                })
        doc.close()
        
        return '\n\n'.join([p['text'] for p in text_content]), text_content
    except Exception as e:
        print(f"PyMuPDF failed: {e}")
        return None, None


def extract_metadata(text):
    """从文本中提取元数据"""
    metadata = {
        'title': '',
        'authors': [],
        'affiliations': [],
        'venue': '',
        'date': ''
    }
    
    # 提取标题（通常在第一行，大写或加粗）
    lines = text.split('\n')
    for i, line in enumerate(lines[:10]):
        line = line.strip()
        if len(line) > 20 and len(line) < 200:
            # 可能是标题
            metadata['title'] = line
            break
    
    # 提取作者（包含 "and" 或逗号分隔的人名）
    author_patterns = [
        r'([A-Z][a-z]+ [A-Z][a-z]+(?: and [A-Z][a-z]+ [A-Z][a-z]+)*)',
        r'([A-Z]\. [A-Z][a-z]+(?:, [A-Z]\. [A-Z][a-z]+)*)',
    ]
    
    for pattern in author_patterns:
        match = re.search(pattern, text[:2000])
        if match:
            authors_str = match.group(1)
            if ' and ' in authors_str:
                metadata['authors'] = [a.strip() for a in authors_str.split(' and ')]
            elif ',' in authors_str:
                metadata['authors'] = [a.strip() for a in authors_str.split(',')]
            else:
                metadata['authors'] = [authors_str]
            break
    
    # 提取 arXiv ID（如果有）
    arxiv_match = re.search(r'arXiv:(\d+\.\d+)', text[:3000])
    if arxiv_match:
        metadata['arxiv_id'] = arxiv_match.group(1)
    
    return metadata


def extract_sections(text):
    """从文本中提取章节"""
    sections = {
        'abstract': '',
        'introduction': '',
        'background': '',
        'method': '',
        'experiments': '',
        'results': '',
        'conclusion': '',
        'references': ''
    }
    
    # 简单的章节分割（基于常见章节标题）
    section_patterns = {
        'abstract': r'(?:ABSTRACT|Abstract)\n(.*?)(?=\n[A-Z]{2,}|$)',
        'introduction': r'(?:INTRODUCTION|Introduction)\n(.*?)(?=\n[A-Z]{2,}|$)',
        'method': r'(?:METHOD|Method|APPROACH|Approach|MODEL|Model)\n(.*?)(?=\n[A-Z]{2,}|$)',
        'experiments': r'(?:EXPERIMENTS|Experiments|EXPERIMENTAL|Experimental)\n(.*?)(?=\n[A-Z]{2,}|$)',
        'conclusion': r'(?:CONCLUSION|Conclusion|DISCUSSION|Discussion)\n(.*?)(?=\n[A-Z]{2,}|$)',
    }
    
    for section_name, pattern in section_patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[section_name] = match.group(1).strip()[:2000]  # 限制长度
    
    return sections


def extract_key_contributions(abstract, introduction):
    """从摘要和引言中提取关键贡献"""
    contributions = []
    
    # 查找贡献陈述的常见模式
    patterns = [
        r'(?:we propose|we present|we introduce|our contribution):?\s*(.*?)(?:\.|\n)',
        r'(?:the main contribution|our contributions) (?:is|are):\s*(.*?)(?:\.|\n)',
        r'(?:(?:first|second|third),? we|we) (?:propose|present|introduce|develop)\s+(.*?)(?:\.|\n)',
    ]
    
    text = abstract + ' ' + introduction[:1000]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match.strip()) > 20 and len(contributions) < 3:
                contributions.append(match.strip())
    
    # 如果没有找到，使用摘要的第一句
    if not contributions and abstract:
        first_sentence = abstract.split('.')[0].strip()
        if len(first_sentence) > 20:
            contributions.append(first_sentence)
    
    return contributions[:3]


def generate_tags(title, abstract, sections):
    """自动生成标签"""
    tags = []
    
    # 从标题和摘要提取关键词
    text = (title + ' ' + abstract).lower()
    
    keyword_mapping = {
        'survey': 'Survey',
        'review': 'Survey',
        'llm': 'LLM',
        'large language model': 'LLM',
        'transformer': 'Transformer',
        'attention': 'Attention',
        'fine-tuning': 'Fine-tuning',
        'prompt': 'Prompt Learning',
        'knowledge distillation': 'Knowledge Distillation',
        'reasoning': 'Reasoning',
        'chain-of-thought': 'Chain-of-Thought',
        'memory': 'AI Memory',
        'agent': 'Agent',
        'multi-agent': 'Multi-Agent',
        'graph': 'Graph Learning',
        'neural': 'Neural Networks',
    }
    
    for keyword, tag in keyword_mapping.items():
        if keyword in text and tag not in tags:
            tags.append(tag)
            if len(tags) >= 8:
                break
    
    return tags


def parse_paper(paper_dir):
    """主解析函数"""
    pdf_path = os.path.join(paper_dir, 'paper.pdf')
    
    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        return None
    
    print(f"Parsing PDF: {pdf_path}")
    
    # 1. 解析 PDF（带 fallback）
    full_text, pages_content = None, None
    
    full_text, pages_content = parse_pdf_with_pdfplumber(pdf_path)
    if not full_text:
        full_text, pages_content = parse_pdf_with_pymupdf(pdf_path)
    
    if not full_text:
        print("All parsers failed!")
        return None
    
    print(f"Extracted {len(full_text)} characters from {len(pages_content)} pages")
    
    # 2. 提取元数据
    metadata = extract_metadata(full_text)
    
    # 3. 提取章节
    sections = extract_sections(full_text)
    
    # 4. 提取关键贡献
    key_contributions = extract_key_contributions(
        sections.get('abstract', ''),
        sections.get('introduction', '')
    )
    
    # 5. 生成标签
    tags = generate_tags(
        metadata.get('title', ''),
        sections.get('abstract', ''),
        sections
    )
    
    # 6. 生成分析结果
    analysis = {
        'metadata': metadata,
        'abstract': sections.get('abstract', ''),
        'sections': sections,
        'figures': [],  # 由 extract_figures.py 填充
        'tables': [],
        'key_contributions': key_contributions,
        'tags': tags,
        'paper_slug': Path(paper_dir).name
    }
    
    # 7. 保存分析 JSON
    output_path = os.path.join(paper_dir, 'analysis.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    print(f"Analysis saved to: {output_path}")
    
    return analysis


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_paper.py <paper_dir>")
        print("Example: python parse_paper.py \"_papers/scott-kd/\"")
        sys.exit(1)
    
    paper_dir = sys.argv[1]
    
    if not os.path.exists(paper_dir):
        print(f"Directory not found: {paper_dir}")
        sys.exit(1)
    
    analysis = parse_paper(paper_dir)
    
    if analysis:
        print("\n=== Parsing Summary ===")
        print(f"Title: {analysis['metadata'].get('title', 'N/A')}")
        print(f"Authors: {', '.join(analysis['metadata'].get('authors', []))}")
        print(f"Contributions: {len(analysis['key_contributions'])}")
        print(f"Tags: {', '.join(analysis['tags'])}")
    else:
        print("\nParsing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
