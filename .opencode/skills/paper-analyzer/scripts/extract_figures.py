#!/usr/bin/env python3
"""
图片提取脚本 - 从 PDF 提取图片并分类

用法:
    python extract_figures.py <pdf_path> <output_dir>

示例:
    python extract_figures.py "_papers/scott-kd/paper.pdf" "_papers/scott-kd/figures/"
"""

import sys
import os
import re
import json
from pathlib import Path


def extract_images_from_pdf(pdf_path, output_dir):
    """从 PDF 提取图片"""
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            img_list = page.get_images(full=True)
            
            for img_idx, img in enumerate(img_list):
                xref = img[0]
                
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image['image']
                    image_ext = base_image['ext']
                    
                    # 过滤太小的图片（可能是图标或装饰）
                    if len(image_bytes) < 1000:
                        continue
                    
                    # 保存图片
                    img_filename = f"fig_{page_num + 1}_{img_idx + 1}.{image_ext}"
                    img_path = os.path.join(output_dir, img_filename)
                    
                    with open(img_path, 'wb') as f:
                        f.write(image_bytes)
                    
                    images.append({
                        'path': img_path,
                        'page': page_num + 1,
                        'index': img_idx + 1,
                        'size': len(image_bytes),
                        'ext': image_ext
                    })
                    
                except Exception as e:
                    print(f"Error extracting image {img_idx}: {e}")
        
        doc.close()
        return images
        
    except Exception as e:
        print(f"PyMuPDF extraction failed: {e}")
        return []


def classify_image(image_info, pdf_text=None):
    """分类图片类型"""
    # 默认分类
    image_type = 'other'
    confidence = 'low'
    suggested_name = f"figure_{image_info['page']}_{image_info['index']}.{image_info['ext']}"
    
    # 基于页面位置推测
    page = image_info['page']
    
    # 第 1-2 页：通常是概述/路线图
    if page <= 2:
        image_type = 'overview'
        confidence = 'medium'
        suggested_name = f"roadmap.{image_info['ext']}"
    
    # 方法章节（通常在 3-6 页）
    elif 3 <= page <= 6:
        image_type = 'method'
        confidence = 'medium'
        suggested_name = f"method.{image_info['ext']}"
    
    # 实验章节（通常在 7-12 页）
    elif 7 <= page <= 12:
        image_type = 'result'
        confidence = 'medium'
        suggested_name = f"results.{image_info['ext']}"
    
    # 基于文件名关键词
    filename_keywords = {
        'overview': ['roadmap', 'overview', 'pipeline', 'workflow'],
        'architecture': ['framework', 'architecture', 'model', 'system'],
        'method': ['method', 'approach', 'algorithm', 'proposed'],
        'result': ['result', 'comparison', 'accuracy', 'performance', 'benchmark'],
        'ablation': ['ablation', 'study', 'analysis'],
        'table': ['table', 'statistic', 'benchmark'],
    }
    
    # 如果有 PDF 文本，尝试匹配关键词
    if pdf_text:
        # 获取图片附近的文本
        text_snippet = pdf_text[image_info['page'] * 1000:(image_info['page'] + 1) * 1000].lower()
        
        for img_type, keywords in filename_keywords.items():
            for keyword in keywords:
                if keyword in text_snippet:
                    image_type = img_type
                    confidence = 'high'
                    suggested_name = f"{keyword}.{image_info['ext']}"
                    break
    
    return {
        'type': image_type,
        'confidence': confidence,
        'suggested_name': suggested_name
    }


def generate_caption(image_type, page):
    """生成图片说明"""
    captions = {
        'overview': '论文内容路线图，展示整体结构和主要章节',
        'architecture': '系统架构图，展示核心组件和数据流',
        'method': '方法示意图，展示提出的核心技术',
        'result': '实验结果对比图，展示性能提升',
        'ablation': '消融实验结果，分析各组件贡献',
        'table': '实验数据表格，对比不同方法的性能',
        'other': '论文插图'
    }
    
    base_caption = captions.get(image_type, '论文插图')
    return f"{base_caption}（第{page}页）"


def extract_figures(pdf_path, output_dir, analysis_json_path=None):
    """主提取函数"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Extracting images from: {pdf_path}")
    print(f"Output directory: {output_dir}")
    
    # 1. 提取图片
    images = extract_images_from_pdf(pdf_path, output_dir)
    print(f"Extracted {len(images)} images")
    
    if not images:
        print("No images found")
        return []
    
    # 2. 加载 PDF 文本（用于分类）
    pdf_text = None
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pdf_text = ''
            for page in pdf.pages[:10]:  # 只读取前 10 页
                text = page.extract_text()
                if text:
                    pdf_text += text
    except:
        pass
    
    # 3. 分类和重命名
    classified_images = []
    for img in images:
        classification = classify_image(img, pdf_text)
        
        caption = generate_caption(classification['type'], img['page'])
        
        classified_images.append({
            'original_path': img['path'],
            'suggested_name': classification['suggested_name'],
            'type': classification['type'],
            'confidence': classification['confidence'],
            'caption': caption,
            'page': img['page'],
            'size': img['size']
        })
    
    # 4. 更新分析 JSON（如果存在）
    if analysis_json_path and os.path.exists(analysis_json_path):
        with open(analysis_json_path, 'r', encoding='utf-8') as f:
            analysis = json.load(f)
        
        # 更新 figures 字段
        analysis['figures'] = classified_images
        
        # 保存
        with open(analysis_json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        print(f"Updated analysis JSON with {len(classified_images)} figures")
    
    # 5. 打印总结
    print("\n=== Figure Extraction Summary ===")
    type_counts = {}
    for img in classified_images:
        img_type = img['type']
        type_counts[img_type] = type_counts.get(img_type, 0) + 1
    
    for img_type, count in type_counts.items():
        print(f"  {img_type}: {count}")
    
    return classified_images


def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_figures.py <pdf_path> <output_dir> [--analysis <json_path>]")
        print("Example: python extract_figures.py \"_papers/scott-kd/paper.pdf\" \"_papers/scott-kd/figures/\"")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    # 解析可选参数
    analysis_json_path = None
    if '--analysis' in sys.argv:
        idx = sys.argv.index('--analysis')
        if idx + 1 < len(sys.argv):
            analysis_json_path = sys.argv[idx + 1]
    
    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)
    
    figures = extract_figures(pdf_path, output_dir, analysis_json_path)
    
    if not figures:
        print("Figure extraction failed or no images found")
        sys.exit(1)
    
    print(f"\nExtraction completed: {len(figures)} figures")


if __name__ == "__main__":
    main()
