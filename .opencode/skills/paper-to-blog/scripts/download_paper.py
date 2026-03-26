#!/usr/bin/env python3
"""
Download arXiv paper PDF and source code.

Usage:
    python download_paper.py <arxiv_id> <output_dir>

Example:
    python download_paper.py 2305.01879 papers/2305.01879
"""

import os
import sys
import subprocess
import tarfile
import shutil


def parse_arxiv_id(url_or_id):
    """Extract arXiv ID from URL or raw ID."""
    if url_or_id.startswith('http'):
        # https://arxiv.org/abs/2305.01879
        # https://arxiv.org/pdf/2305.01879.pdf
        parts = url_or_id.rstrip('/').split('/')
        arxiv_id = parts[-1].replace('.pdf', '')
        return arxiv_id
    else:
        # Raw ID like 2305.01879
        return url_or_id


def download_paper(url_or_id, output_dir):
    """Download arXiv paper PDF and source code."""
    arxiv_id = parse_arxiv_id(url_or_id)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download PDF
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    pdf_path = os.path.join(output_dir, "paper.pdf")
    print(f"Downloading PDF from {pdf_url}...")
    subprocess.run(['curl', '-L', '-o', pdf_path, pdf_url], check=True)
    
    # Download source code
    source_url = f"https://arxiv.org/e-print/{arxiv_id}"
    source_path = os.path.join(output_dir, "source.tar.gz")
    print(f"Downloading source from {source_url}...")
    subprocess.run(['curl', '-L', '-o', source_path, source_url], check=True)
    
    # Extract source code
    print("Extracting source code...")
    with tarfile.open(source_path, 'r:gz') as tar:
        tar.extractall(path=output_dir)
    
    print(f"✓ Paper downloaded to {output_dir}")
    return arxiv_id, output_dir


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python download_paper.py <arxiv_id_or_url> <output_dir>")
        sys.exit(1)
    
    url_or_id = sys.argv[1]
    output_dir = sys.argv[2]
    
    arxiv_id, output_dir = download_paper(url_or_id, output_dir)
    print(f"\nArXiv ID: {arxiv_id}")
    print(f"Output directory: {output_dir}")
