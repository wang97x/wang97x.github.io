#!/usr/bin/env python3
"""
论文分析主流程 - 一键完成下载、解析、图片提取

用法:
    python analyze_paper.py <input_source> --categories "<cat1>,<cat2>" [--next paper-to-blog]

示例:
    python analyze_paper.py "https://arxiv.org/abs/2305.01879" --categories "论文阅读，知识蒸馏"
    python analyze_paper.py "https://arxiv.org/abs/2305.01879" --categories "论文阅读，知识蒸馏" --next paper-to-blog
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """运行命令并检查错误"""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        print(f"Error: {result.stderr}")
        return False
    
    print(f"✅ Success: {description}")
    return True


def get_script_dir():
    """获取脚本所在目录"""
    return Path(__file__).parent


def analyze_paper(input_source, categories, output_dir=None, run_blog_converter=False):
    """主分析流程"""
    
    script_dir = get_script_dir()
    
    # 1. 生成输出目录名
    if not output_dir:
        # 从输入生成 slug
        from datetime import datetime
        date_prefix = datetime.now().strftime('%Y-%m-%d')
        
        # 简单提取标题
        if 'arxiv.org' in input_source:
            arxiv_id = input_source.split('/')[-1].split('.')[0]
            slug = f"arxiv-{arxiv_id}"
        elif 'github.com' in input_source:
            parts = input_source.rstrip('/').split('/')
            slug = f"github-{parts[-1]}"
        else:
            slug = f"paper-{date_prefix}"
        
        output_dir = f"_papers/{slug}"
    
    print(f"\n📁 Output directory: {output_dir}")
    
    # 2. 下载论文
    download_script = script_dir / 'download_paper.py'
    if not run_command(
        [sys.executable, str(download_script), input_source, output_dir],
        "Download paper"
    ):
        return None, None
    
    # 3. 解析论文
    parse_script = script_dir / 'parse_paper.py'
    if not run_command(
        [sys.executable, str(parse_script), output_dir],
        "Parse paper content"
    ):
        return None, None
    
    # 4. 提取图片
    extract_script = script_dir / 'extract_figures.py'
    pdf_path = os.path.join(output_dir, 'paper.pdf')
    figures_dir = os.path.join(output_dir, 'figures')
    analysis_path = os.path.join(output_dir, 'analysis.json')
    
    if not run_command(
        [sys.executable, str(extract_script), pdf_path, figures_dir, '--analysis', analysis_path],
        "Extract figures"
    ):
        print("⚠️ Figure extraction failed, continuing without figures")
    
    # 5. 读取分析结果
    with open(analysis_path, 'r', encoding='utf-8') as f:
        analysis = json.load(f)
    
    print(f"\n{'='*60}")
    print("✅ Paper analysis completed!")
    print('='*60)
    print(f"\nOutput files:")
    print(f"  - Analysis JSON: {analysis_path}")
    print(f"  - PDF: {os.path.join(output_dir, 'paper.pdf')}")
    print(f"  - Figures: {figures_dir}/")
    
    # 6. 可选：调用 paper-to-blog
    if run_blog_converter:
        print(f"\n{'='*60}")
        print("Step: Convert to blog post")
        print('='*60)
        
        blog_script = Path(script_dir.parent / 'paper-to-blog' / 'scripts' / 'convert_to_blog.py')
        
        if blog_script.exists():
            run_command(
                [sys.executable, str(blog_script), analysis_path, '--categories', categories, '--output', '_posts/'],
                "Convert to blog"
            )
        else:
            print(f"⚠️ Blog converter script not found: {blog_script}")
    
    return output_dir, analysis


def main():
    parser = argparse.ArgumentParser(description='Analyze paper and optionally convert to blog')
    parser.add_argument('input_source', help='arXiv URL, GitHub URL, PDF path, or paper title')
    parser.add_argument('--categories', default='论文阅读，通用', help='Categories like "论文阅读，知识蒸馏"')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--next', choices=['paper-to-blog'], help='Next step after analysis')
    parser.add_argument('--auto-cleanup', action='store_true', help='Auto cleanup temp files after blog generation')
    
    args = parser.parse_args()
    
    run_blog = (args.next == 'paper-to-blog')
    
    output_dir, analysis = analyze_paper(
        args.input_source,
        args.categories,
        args.output,
        run_blog
    )
    
    if not output_dir:
        print("\n❌ Analysis failed!")
        sys.exit(1)
    
    # 7. 询问清理
    if run_blog and args.auto_cleanup:
        print(f"\n{'='*60}")
        response = input(f"Delete temporary files in {output_dir}? [y/N]: ")
        if response.lower() == 'y':
            import shutil
            try:
                shutil.rmtree(output_dir)
                print(f"✅ Cleaned up: {output_dir}")
            except Exception as e:
                print(f"⚠️ Failed to cleanup: {e}")


if __name__ == "__main__":
    # 需要导入 json
    import json
    main()
