import pdfplumber
import json

with pdfplumber.open('Survey_on_AI_Memory.pdf') as pdf:
    print(f'Total pages: {len(pdf.pages)}')
    
    all_text = []
    # 提取所有页面的内容
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            all_text.append({
                'page': i + 1,
                'text': text
            })
            print(f'Extracted page {i+1}')
    
    # 保存到 JSON 文件
    with open('pdf_content.json', 'w', encoding='utf-8') as f:
        json.dump(all_text, f, ensure_ascii=False, indent=2)
    
    print(f'Saved content from {len(all_text)} pages to pdf_content.json')
