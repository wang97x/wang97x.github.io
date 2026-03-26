#!/usr/bin/env python3
"""
Parse LaTeX source files to extract paper content.

Usage:
    python parse_latex.py <paper_dir>

Example:
    python parse_latex.py papers/2305.01879
"""

import os
import sys
import re


def read_tex_file(filepath):
    """Read LaTeX file and return content."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def extract_text_from_tex(content):
    """Extract plain text from LaTeX content."""
    # Remove comments
    content = re.sub(r'%.*$', '', content, flags=re.MULTILINE)
    
    # Remove LaTeX commands (keep text)
    content = re.sub(r'\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^}]*\})?', ' ', content)
    content = re.sub(r'[{}]', '', content)
    
    # Clean up whitespace
    content = re.sub(r'\s+', ' ', content).strip()
    
    return content


def find_section_files(paper_dir):
    """Find section .tex files in the paper directory."""
    sections_dir = os.path.join(paper_dir, 'sections')
    section_files = {}
    
    if os.path.exists(sections_dir):
        for f in sorted(os.listdir(sections_dir)):
            if f.endswith('.tex'):
                filepath = os.path.join(sections_dir, f)
                # Extract section name from filename
                match = re.search(r'\d+_(.+)\.tex', f)
                if match:
                    section_name = match.group(1)
                    section_files[section_name] = filepath
    
    return section_files


def extract_metadata(main_tex_path):
    """Extract paper metadata from main.tex."""
    content = read_tex_file(main_tex_path)
    
    metadata = {
        'title': '',
        'authors': [],
        'abstract': ''
    }
    
    # Extract title
    title_match = re.search(r'\\title\{([^}]+)\}', content)
    if title_match:
        metadata['title'] = title_match.group(1).strip()
    
    # Extract authors
    author_match = re.search(r'\\author\{([^}]+)\}', content)
    if author_match:
        authors_raw = author_match.group(1)
        # Simple extraction (may need refinement)
        metadata['authors'] = [a.strip() for a in re.split(r'\\And|,', authors_raw) if a.strip()]
    
    # Extract abstract
    abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', content, re.DOTALL)
    if abstract_match:
        metadata['abstract'] = extract_text_from_tex(abstract_match.group(1))
    
    return metadata


def parse_paper(paper_dir):
    """Parse paper and extract structured content."""
    content = {
        'metadata': {},
        'sections': {}
    }
    
    # Extract metadata from main.tex
    main_tex = os.path.join(paper_dir, 'main.tex')
    if os.path.exists(main_tex):
        content['metadata'] = extract_metadata(main_tex)
    
    # Extract sections
    section_files = find_section_files(paper_dir)
    for section_name, filepath in section_files.items():
        tex_content = read_tex_file(filepath)
        text_content = extract_text_from_tex(tex_content)
        content['sections'][section_name] = {
            'raw_tex': tex_content,
            'text': text_content
        }
    
    return content


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python parse_latex.py <paper_dir>")
        sys.exit(1)
    
    paper_dir = sys.argv[1]
    
    content = parse_paper(paper_dir)
    
    print("Paper Metadata:")
    print(f"  Title: {content['metadata']['title']}")
    print(f"  Authors: {', '.join(content['metadata']['authors'][:3])} et al." if len(content['metadata']['authors']) > 3 else f"  Authors: {', '.join(content['metadata']['authors'])}")
    
    print("\nSections found:")
    for name in content['sections'].keys():
        print(f"  - {name}")
    
    print("\nAbstract:")
    print(f"  {content['metadata']['abstract'][:200]}...")
