#!/usr/bin/env python3
"""
论文下载脚本 - 支持多来源和 fallback 策略

用法:
    python download_paper.py <input_source> <output_dir>

示例:
    python download_paper.py "https://arxiv.org/abs/2305.01879" "_papers/scott-kd"
    python download_paper.py "https://github.com/user/repo" "_papers/repo-paper"
    python download_paper.py "paper.pdf" "_papers/local-pdf"
"""

import sys
import os
import re
import time
import urllib.request
import ssl
from pathlib import Path
from urllib.parse import urlparse, parse_qs


def extract_arxiv_id(url):
    """从 arXiv URL 提取 ID"""
    patterns = [
        r'arxiv\.org/abs/(\d+\.\d+)',
        r'arxiv\.org/pdf/(\d+\.\d+)',
        r'arXiv:(\d+\.\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def download_with_retry(url, output_path, max_retries=3, min_size=1024):
    """带重试的下载，验证文件大小"""
    # 禁用 SSL 验证（用于某些网站）
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    for attempt in range(max_retries):
        try:
            print(f"Downloading: {url} (attempt {attempt + 1}/{max_retries})")
            urllib.request.urlretrieve(url, output_path)
            
            # 验证文件大小
            size = os.path.getsize(output_path)
            if size < min_size:
                print(f"Warning: File too small ({size} bytes), retrying...")
                os.remove(output_path)
                time.sleep(2 ** attempt)  # 指数退避
                continue
            
            # 验证 PDF 文件头
            if output_path.endswith('.pdf'):
                with open(output_path, 'rb') as f:
                    header = f.read(5)
                    if not header.startswith(b'%PDF'):
                        print(f"Warning: Invalid PDF header, retrying...")
                        os.remove(output_path)
                        time.sleep(2 ** attempt)
                        continue
            
            print(f"Downloaded successfully: {output_path} ({size} bytes)")
            return True
            
        except Exception as e:
            print(f"Download failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return False
    
    return False


def download_arxiv(arxiv_id, output_dir):
    """从 arXiv 下载论文"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载 PDF
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    pdf_path = os.path.join(output_dir, "paper.pdf")
    success = download_with_retry(pdf_url, pdf_path)
    
    if not success:
        print("Failed to download PDF from arXiv")
        return False
    
    # 尝试下载 LaTeX 源码（用于提取图片和表格）
    src_url = f"https://arxiv.org/e-print/{arxiv_id}"
    src_path = os.path.join(output_dir, "source.tar.gz")
    download_with_retry(src_url, src_path, max_retries=2, min_size=100)
    
    return True


def download_github(repo_url, output_dir):
    """从 GitHub 仓库提取信息"""
    # 提取 repo 信息
    match = re.search(r'github\.com/([^/]+)/([^/]+)', repo_url)
    if not match:
        print("Invalid GitHub URL")
        return False
    
    owner, repo = match.groups()
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载 README
    readme_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md"
    readme_path = os.path.join(output_dir, "README.md")
    
    if not download_with_retry(readme_url, readme_path, min_size=50):
        # 尝试 master 分支
        readme_url = f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.md"
        download_with_retry(readme_url, readme_path, min_size=50)
    
    # 查找 releases 中的论文 PDF
    # 这里可以扩展：调用 GitHub API 查找 releases 或 wiki 中的论文链接
    
    print(f"Extracted info from GitHub: {owner}/{repo}")
    return True


def download_pdf_direct(url, output_dir):
    """直接下载 PDF 文件"""
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "paper.pdf")
    return download_with_retry(url, pdf_path)


def main():
    if len(sys.argv) < 3:
        print("Usage: python download_paper.py <input_source> <output_dir>")
        print("Examples:")
        print('  python download_paper.py "https://arxiv.org/abs/2305.01879" "_papers/scott-kd"')
        print('  python download_paper.py "https://github.com/user/repo" "_papers/repo-paper"')
        sys.exit(1)
    
    input_source = sys.argv[1]
    output_dir = sys.argv[2]
    
    # 1. 检测输入类型
    arxiv_id = extract_arxiv_id(input_source)
    
    if arxiv_id:
        print(f"Detected arXiv ID: {arxiv_id}")
        success = download_arxiv(arxiv_id, output_dir)
    elif input_source.startswith("https://github.com"):
        print("Detected GitHub repository")
        success = download_github(input_source, output_dir)
    elif input_source.endswith(".pdf"):
        print("Detected PDF file")
        if input_source.startswith("http"):
            success = download_pdf_direct(input_source, output_dir)
        elif os.path.exists(input_source):
            # 本地文件，复制到输出目录
            os.makedirs(output_dir, exist_ok=True)
            import shutil
            shutil.copy(input_source, os.path.join(output_dir, "paper.pdf"))
            success = True
        else:
            print(f"PDF file not found: {input_source}")
            success = False
    elif input_source.startswith("http"):
        # 其他网页，尝试抓取并查找 PDF 链接
        print(f"Webpage detected, will try to extract PDF link: {input_source}")
        # TODO: 实现网页抓取和 PDF 链接提取
        success = download_pdf_direct(input_source, output_dir)
    else:
        print(f"Unknown input format: {input_source}")
        print("Supported formats:")
        print("  - arXiv URL: https://arxiv.org/abs/XXXX.XXXXX")
        print("  - GitHub URL: https://github.com/owner/repo")
        print("  - PDF URL or local path")
        print("  - Project homepage URL")
        success = False
    
    if success:
        print(f"\nDownload completed: {output_dir}")
    else:
        print(f"\nDownload failed: {output_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
