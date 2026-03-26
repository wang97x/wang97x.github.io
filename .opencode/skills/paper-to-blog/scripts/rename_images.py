#!/usr/bin/env python3
"""
图片重命名脚本 - 根据分类重命名图片并生成说明

用法:
    python rename_images.py <figures_dir> <output_dir> --analysis <analysis.json>

示例:
    python rename_images.py "_papers/slug/figures" "assets/images/slug" --analysis "_papers/slug/analysis.json"
"""

import sys
import os
import json
import shutil
from pathlib import Path


def load_classification_rules():
    """加载分类规则"""
    rules_path = Path(__file__).parent.parent.parent / 'paper-analyzer' / 'config' / 'classification-rules.json'
    
    if rules_path.exists():
        with open(rules_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # 默认规则
    return {
        "naming_convention": {
            "overview": "roadmap",
            "architecture": "framework",
            "method": "method",
            "result": "results",
            "ablation": "ablation",
            "table": "table",
            "other": "figure"
        },
        "captions": {
            "overview": "论文内容路线图，展示整体结构和主要章节",
            "architecture": "系统架构图，展示核心组件和数据流",
            "method": "方法示意图，展示提出的核心技术",
            "result": "实验结果对比图，展示性能提升",
            "ablation": "消融实验结果，分析各组件贡献",
            "table": "实验数据表格，对比不同方法的性能",
            "other": "论文插图"
        }
    }


def rename_images(figures_dir, output_dir, analysis_json_path):
    """重命名图片并生成说明"""
    print(f"\nRenaming images...")
    print(f"  Source: {figures_dir}")
    print(f"  Output: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载分析 JSON
    with open(analysis_json_path, 'r', encoding='utf-8') as f:
        analysis = json.load(f)
    
    # 加载分类规则
    rules = load_classification_rules()
    naming = rules.get('naming_convention', {})
    captions = rules.get('captions', {})
    
    # 处理图片
    figures = analysis.get('figures', [])
    renamed_figures = []
    used_names = {}  # 避免重名
    
    for i, fig in enumerate(figures):
        src_path = fig.get('path', '')
        img_type = fig.get('type', 'other')
        
        if not src_path or not os.path.exists(src_path):
            print(f"  ⚠️  File not found: {src_path}")
            continue
        
        # 生成新名称
        base_name = naming.get(img_type, 'figure')
        
        # 避免重名
        if base_name in used_names:
            used_names[base_name] += 1
            new_name = f"{base_name}_{used_names[base_name]}.png"
        else:
            used_names[base_name] = 1
            new_name = f"{base_name}.png"
        
        # 复制并重命名
        dst_path = os.path.join(output_dir, new_name)
        shutil.copy2(src_path, dst_path)
        
        # 生成说明
        caption = fig.get('caption', captions.get(img_type, '论文插图'))
        page = fig.get('page', 0)
        if page:
            caption = f"{caption}（第{page}页）"
        
        renamed_figures.append({
            'original_path': src_path,
            'new_path': dst_path,
            'new_name': new_name,
            'type': img_type,
            'caption': caption
        })
        
        print(f"  ✅  {os.path.basename(src_path)} → {new_name} ({img_type})")
    
    # 更新分析 JSON
    analysis['renamed_figures'] = renamed_figures
    analysis['image_dir'] = output_dir
    
    # 保存更新后的分析 JSON
    with open(analysis_json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅  Renamed {len(renamed_figures)} images")
    print(f"    Updated analysis JSON: {analysis_json_path}")
    
    return renamed_figures


def generate_image_captions(renamed_figures):
    """生成图片说明 Markdown"""
    captions = []
    
    for i, fig in enumerate(renamed_figures, 1):
        caption_md = f"![图{i}](/assets/images/{Path(fig['new_path']).parent.name}/{fig['new_name']})\n"
        caption_md += f"*图{i}: {fig['caption']}*\n"
        captions.append(caption_md)
    
    return '\n'.join(captions)


def main():
    if len(sys.argv) < 4:
        print("Usage: python rename_images.py <figures_dir> <output_dir> --analysis <json_path>")
        print("Example:")
        print('  python rename_images.py "_papers/slug/figures" "assets/images/slug" --analysis "_papers/slug/analysis.json"')
        sys.exit(1)
    
    figures_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # 解析参数
    analysis_json_path = None
    if '--analysis' in sys.argv:
        idx = sys.argv.index('--analysis')
        if idx + 1 < len(sys.argv):
            analysis_json_path = sys.argv[idx + 1]
    
    if not analysis_json_path:
        print("Error: --analysis parameter is required")
        sys.exit(1)
    
    if not os.path.exists(analysis_json_path):
        print(f"Error: Analysis JSON not found: {analysis_json_path}")
        sys.exit(1)
    
    # 重命名图片
    renamed_figures = rename_images(figures_dir, output_dir, analysis_json_path)
    
    if not renamed_figures:
        print("No images to rename")
        sys.exit(1)
    
    # 生成说明示例
    print("\n=== Generated Captions ===")
    print(generate_image_captions(renamed_figures))


if __name__ == "__main__":
    main()
