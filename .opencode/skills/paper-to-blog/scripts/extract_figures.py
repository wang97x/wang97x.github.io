#!/usr/bin/env python3
"""
Extract figures from arXiv paper source package.

Usage:
    python extract_figures.py <paper_dir> <output_dir>

Example:
    python extract_figures.py papers/2305.01879 assets/images/SCOTT
"""

import os
import sys
import shutil
import fitz  # PyMuPDF


# Key figure patterns to extract (priority order)
KEY_FIGURE_PATTERNS = [
    'pipeline',           # Overall framework
    'framework',          # Framework
    'architecture',       # Architecture
    'model',              # Model architecture
    'method',             # Method overview
    'approach',           # Approach
    'overview',           # Overview
    'contrastive',        # Contrastive decoding
    'counterfactual',     # Counterfactual
    'training',           # Training process
    'teacher',            # Teacher results
    'student',            # Student results
    'acc_las',            # Accuracy vs LAS
    'perf_change',        # Performance change
    'ablation',           # Ablation study
    'student_size',       # Student size ablation
    'vacuous',            # Vacuous rationale
    'case_study',         # Case study
]

# Maximum number of figures to extract
MAX_FIGURES = 8


def find_pdf_figures(paper_dir):
    """Find PDF figure files in possible figure directories."""
    # Support multiple possible directory names
    possible_dirs = ['figures', 'images', 'figs', 'img', 'pics']
    
    figures_dir = None
    for dir_name in possible_dirs:
        test_dir = os.path.join(paper_dir, dir_name)
        if os.path.exists(test_dir):
            figures_dir = test_dir
            break
    
    if not figures_dir:
        return [], None
    
    pdf_files = []
    for f in os.listdir(figures_dir):
        if f.endswith('.pdf'):
            pdf_files.append(f)
    
    return pdf_files, figures_dir


def prioritize_figures(pdf_files):
    """Sort figures by priority based on naming patterns."""
    def get_priority(filename):
        name_lower = filename.lower()
        for i, pattern in enumerate(KEY_FIGURE_PATTERNS):
            if pattern in name_lower:
                return i
        return len(KEY_FIGURE_PATTERNS)
    
    return sorted(pdf_files, key=get_priority)


def convert_pdf_to_png(pdf_path, output_path, zoom=2):
    """Convert first page of PDF to PNG."""
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        pix.save(output_path)
        doc.close()
        return True
    except Exception as e:
        print(f"  [Warning] Error converting {pdf_path}: {e}")
        return False


def extract_figures(paper_dir, output_dir):
    """Extract key figures from paper source package."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF figures (supports multiple directory names)
    pdf_files, figures_dir = find_pdf_figures(paper_dir)
    
    if not figures_dir:
        print("[Warning] No figures/images directory found")
        return []
    
    print(f"Found {len(pdf_files)} PDF figures in {figures_dir}")
    
    if not pdf_files:
        print("[Warning] No PDF figures found!")
        return []
    
    # Prioritize figures
    prioritized = prioritize_figures(pdf_files)
    print(f"Prioritized figure order:")
    for i, f in enumerate(prioritized[:MAX_FIGURES]):
        print(f"  {i+1}. {f}")
    
    # Convert top figures to PNG
    converted = []
    for pdf_name in prioritized[:MAX_FIGURES]:
        pdf_path = os.path.join(figures_dir, pdf_name)
        png_name = pdf_name.replace('.pdf', '.png')
        png_path = os.path.join(output_dir, png_name)
        
        print(f"Converting {pdf_name}...")
        if convert_pdf_to_png(pdf_path, png_path):
            converted.append(png_name)
            print(f"  [OK] {png_name}")
    
    print(f"\n[OK] Extracted {len(converted)} figures to {output_dir}")
    return converted


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python extract_figures.py <paper_dir> <output_dir>")
        sys.exit(1)
    
    paper_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    converted = extract_figures(paper_dir, output_dir)
    
    print(f"\nConverted figures:")
    for f in converted:
        print(f"  - {f}")
