#!/usr/bin/env python3
"""
博客验证脚本 - 发布前检查

用法:
    python validate_blog.py <blog_post_path>

检查项目:
- Frontmatter 格式
- 分类/标签分隔符
- 图片路径存在性
- 图片说明格式
- 公式格式
"""

import sys
import os
import re
import yaml
from pathlib import Path


def load_blog_post(path):
    """加载博客文章"""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 分离 Frontmatter
    if not content.startswith('---'):
        return None, None, "Missing Frontmatter start '---'"
    
    parts = content.split('---', 2)
    if len(parts) < 3:
        return None, None, "Invalid Frontmatter format"
    
    frontmatter_str = parts[1].strip()
    body = parts[2].strip()
    
    try:
        frontmatter = yaml.safe_load(frontmatter_str)
    except yaml.YAMLError as e:
        return None, None, f"YAML parsing error: {e}"
    
    return frontmatter, body, None


def check_frontmatter(frontmatter, path):
    """检查 Frontmatter"""
    errors = []
    warnings = []
    
    # 必填字段
    required_fields = ['layout', 'title', 'date', 'categories', 'tags', 'image']
    for field in required_fields:
        if field not in frontmatter:
            errors.append(f"Missing required field: {field}")
    
    # 检查 layout
    if frontmatter.get('layout') != 'post':
        warnings.append(f"Layout should be 'post', got: {frontmatter.get('layout')}")
    
    # 检查 categories 格式
    categories = frontmatter.get('categories', [])
    if isinstance(categories, str):
        if '，' in categories:
            errors.append("Categories should use English comma ',' not Chinese comma '，'")
    elif not isinstance(categories, list):
        errors.append(f"Categories should be a list, got: {type(categories)}")
    
    # 检查 tags 格式
    tags = frontmatter.get('tags', [])
    if isinstance(tags, str):
        if '，' in tags:
            errors.append("Tags should use English comma ',' not Chinese comma '，'")
        if len(tags.split(',')) > 8:
            warnings.append(f"Too many tags (>8): {len(tags.split(','))}")
    elif isinstance(tags, list):
        if len(tags) > 8:
            warnings.append(f"Too many tags (>8): {len(tags)}")
    
    # 检查 image 路径
    image = frontmatter.get('image', '')
    if image and not image.startswith('/assets/images/'):
        errors.append(f"Image path should start with '/assets/images/', got: {image}")
    
    # 检查 image 文件存在
    if image:
        # 转换为绝对路径
        blog_dir = Path(path).parent.parent
        image_path = blog_dir / image.lstrip('/')
        if not image_path.exists():
            errors.append(f"Image file not found: {image_path}")
    
    return errors, warnings


def check_body(body, path):
    """检查正文"""
    errors = []
    warnings = []
    
    # 检查图片说明格式（斜体）
    image_pattern = r'!\[.*?\]\(.*?\)'
    images = re.findall(image_pattern, body)
    
    for img in images:
        # 检查后面是否有斜体说明
        img_pos = body.find(img)
        after_img = body[img_pos:img_pos + 200]
        
        # 斜体格式：*图 X：...*
        if not re.search(r'\*图\s*\d+[:：].*?\*', after_img):
            warnings.append(f"Image missing italic caption: {img[:50]}")
    
    # 检查公式格式
    inline_math = re.findall(r'\$[^$]+\$', body)
    display_math = re.findall(r'\$\$[^$]+\$\$', body)
    
    # 检查是否有未闭合的公式
    if body.count('$$') % 2 != 0:
        errors.append("Unclosed display math formula ($$)")
    
    # 检查 AI 分析方法亮点部分
    if '## AI 分析方法亮点' not in body:
        warnings.append("Missing '## AI 分析方法亮点' section")
    else:
        # 检查三个维度
        highlights_section = body.split('## AI 分析方法亮点')[1].split('##')[0]
        required_dims = ['问题定位精准', '方法创新', '实用性强']
        for dim in required_dims:
            if dim not in highlights_section:
                warnings.append(f"Missing highlight dimension: {dim}")
    
    # 检查参考链接
    if '## 参考链接' not in body:
        warnings.append("Missing '## 参考链接' section")
    
    # 检查字数
    word_count = len(body)
    if word_count < 1500:
        warnings.append(f"Content too short (<1500 chars): {word_count}")
    elif word_count > 5000:
        warnings.append(f"Content too long (>5000 chars): {word_count}")
    
    return errors, warnings


def validate_blog_post(path):
    """主验证函数"""
    print(f"\nValidating: {path}\n")
    
    # 1. 加载
    frontmatter, body, error = load_blog_post(path)
    if error:
        print(f"[ERROR] {error}")
        return False
    
    # 2. 检查 Frontmatter
    fm_errors, fm_warnings = check_frontmatter(frontmatter, path)
    
    # 3. 检查正文
    body_errors, body_warnings = check_body(body, path)
    
    # 4. 打印结果
    all_errors = fm_errors + body_errors
    all_warnings = fm_warnings + body_warnings
    
    print("=== Validation Results ===\n")
    
    if all_errors:
        print("[ERROR] ERRORS:")
        for err in all_errors:
            print(f"  - {err}")
        print()
    
    if all_warnings:
        print("[WARN] WARNINGS:")
        for warn in all_warnings:
            print(f"  - {warn}")
        print()
    
    if not all_errors and not all_warnings:
        print("[OK] All checks passed!")
    
    # 5. 总结
    print(f"Total: {len(all_errors)} errors, {len(all_warnings)} warnings")
    
    return len(all_errors) == 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_blog.py <blog_post_path>")
        print("Example: python validate_blog.py \"_posts/2024-03-26-scott-kd.md\"")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)
    
    success = validate_blog_post(path)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
