#!/usr/bin/env python3
"""
一键转换流程 - 从 arXiv 链接到博客文章

用法:
    python convert_pipeline.py <arxiv_url> --categories "<cat1>,<cat2>"

示例:
    python convert_pipeline.py "https://arxiv.org/abs/2305.01879" --categories "论文阅读，知识蒸馏"
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def run_step(cmd, description):
    """运行步骤并检查错误"""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] Failed: {description}")
        print(f"Error: {result.stderr}")
        return False
    
    print(f"[OK] Success: {description}")
    return True


def main():
    parser = argparse.ArgumentParser(description='One-click paper to blog conversion')
    parser.add_argument('input', help='arXiv URL or paper source')
    parser.add_argument('--categories', default='论文阅读，通用', help='Categories like "论文阅读，知识蒸馏"')
    parser.add_argument('--output', default='_posts/', help='Output directory for blog post')
    parser.add_argument('--papers-dir', default='_papers', help='Directory for paper analysis files')
    parser.add_argument('--cleanup', action='store_true', help='Cleanup temp files after conversion')
    
    args = parser.parse_args()
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    analyzer_dir = script_dir.parent / 'paper-analyzer' / 'scripts'
    
    print(f"\n{'='*60}")
    print("Paper to Blog - One-Click Conversion")
    print('='*60)
    print(f"Input: {args.input}")
    print(f"Categories: {args.categories}")
    print(f"Output: {args.output}")
    
    # Step 1: 下载论文
    download_script = analyzer_dir / 'download_paper.py'
    if not download_script.exists():
        print(f"[ERROR] Download script not found: {download_script}")
        sys.exit(1)
    
    # 生成输出目录名
    if 'arxiv.org' in args.input:
        arxiv_id = args.input.split('/')[-1].split('.')[0]
        paper_slug = f"arxiv-{arxiv_id}"
    else:
        paper_slug = f"paper-{len(os.listdir(args.papers_dir)) if os.path.exists(args.papers_dir) else 0}"
    
    paper_dir = os.path.join(args.papers_dir, paper_slug)
    
    if not run_step(
        [sys.executable, str(download_script), args.input, paper_dir],
        "Download paper"
    ):
        sys.exit(1)
    
    # Step 2: 解析论文
    parse_script = analyzer_dir / 'parse_paper.py'
    if not run_step(
        [sys.executable, str(parse_script), paper_dir],
        "Parse paper content"
    ):
        sys.exit(1)
    
    # Step 3: 提取图片
    extract_script = analyzer_dir / 'extract_figures.py'
    pdf_path = os.path.join(paper_dir, 'paper.pdf')
    figures_dir = os.path.join(paper_dir, 'figures')
    analysis_path = os.path.join(paper_dir, 'analysis.json')
    
    run_step(
        [sys.executable, str(extract_script), pdf_path, figures_dir, '--analysis', analysis_path],
        "Extract figures"
    )
    
    # Step 4: 转换博客
    convert_script = script_dir / 'convert_to_blog.py'
    if not run_step(
        [sys.executable, str(convert_script), analysis_path, '--categories', args.categories, '--output', args.output],
        "Convert to blog"
    ):
        sys.exit(1)
    
    # Step 5: 验证博客
    validate_script = script_dir / 'validate_blog.py'
    if validate_script.exists():
        # 查找生成的博客文件
        import glob
        blog_files = glob.glob(os.path.join(args.output, f"*-{paper_slug}*.md"))
        if blog_files:
            blog_path = blog_files[0]
            run_step(
                [sys.executable, str(validate_script), blog_path],
                "Validate blog post"
            )
    
    # Step 6: 清理（可选）
    if args.cleanup:
        print(f"\n{'='*60}")
        response = input(f"Delete temporary files in {paper_dir}? [y/N]: ")
        if response.lower() == 'y':
            import shutil
            try:
                shutil.rmtree(paper_dir)
                print(f"[OK] Cleaned up: {paper_dir}")
            except Exception as e:
                print(f"[WARN] Failed to cleanup: {e}")
    
    print(f"\n{'='*60}")
    print("[OK] Conversion completed successfully!")
    print('='*60)
    print(f"\nOutput files:")
    print(f"  - Blog post: {args.output}")
    print(f"  - Analysis: {analysis_path}")
    print(f"  - Images: assets/images/{paper_slug}/")
    
    if not args.cleanup:
        print(f"\nNote: Temporary files kept in {paper_dir}")
        print("Use --cleanup flag to auto-remove after conversion")


if __name__ == "__main__":
    main()
