import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random
import json
import re 


INPUT_FILE = 'all_question_links.csv' 
OUTPUT_JSON = 'du_lieu_luat_dataset.json'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

def extract_legal_reference(text):

    
    clean_text = re.sub(r'\s+', ' ', text).strip()
    references = []
    doc_types = r"(?:Luật|Bộ luật|Nghị định|Thông tư|Quyết định|Nghị quyết|Hiến pháp|Pháp lệnh)"
    valid_number = r"(?:số\s+)?[0-9]+|[IVX]+|[a-đA-Đ]\b"
    unit_name = r"(?:Điều|Khoản|Điểm|Mục|Phần|Chương|Phụ lục|Tiểu mục)"
    valid_unit_block = rf"{unit_name}\s+(?:{valid_number})"

    pattern = (
        r'\b('                                     
        r'(?:' + valid_unit_block + r'[\s,]+)+?'   
        r')'                                      
        r'(?:và\s+)?'                             
        r'(?:của|tại|thuộc|trong|theo|về)?\s*'    
        r'(' + doc_types + r')'                    
        r'(?:\s+số)?\s+'                          
        r'([0-9]+/[\w\-/]+|[^0-9\.\,]*?\d{4})'    
    )

    matches = re.finditer(pattern, clean_text, re.IGNORECASE)
    
    for match in matches:
        raw_ref = match.group(1).strip()
        doc_type = match.group(2).strip()
        doc_id = match.group(3).strip()

        clean_prefix = re.sub(r'[,]+$', '', raw_ref).strip()
        clean_prefix = re.sub(r'\s+(của|tại|trong|thuộc|theo)$', '', clean_prefix).strip()
        
        full_ref = f"{clean_prefix} {doc_type} {doc_id}"
        references.append(full_ref)

    if not references:
        simple = r'(Điều\s+\d+)\s+(?:của|tại)?\s*(Luật|Bộ luật)\s+([A-ZÀ-Ỹ][\w\s]+?)(?=\s+(?:bởi|tại|theo|quy định|thì|năm|\.|\,|$))'
        matches_simple = re.findall(simple, clean_text)
        for m in matches_simple:
            references.append(f"{m[0]} {m[1]} {m[2]}".strip())

    return list(set(references))
def get_detail_content_json(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        title_tag = soup.find('h1', class_='the-article-title')
        question = title_tag.get_text(strip=True) if title_tag else ""
        
        body_div = soup.find('div', class_='the-article-body')
        
        answer = ""
        extracted_refs = []
        
        if body_div:
            p_tags = body_div.find_all('p')
            if len(p_tags) >= 2:
                p_tags[-1].decompose()
                p_tags[-2].decompose()
            elif len(p_tags) == 1:
                p_tags[-1].decompose()

            answer = body_div.get_text(separator='\n', strip=True)
            extracted_refs = extract_legal_reference(answer)
            
        return {
            "question": question,
            "answer": answer,
            "reference": extracted_refs, 
            "url": url
        }

    except Exception as e:
        print(f"Lỗi URL {url}: {e}")
        return None

def main():
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Đã đọc {len(df)} link.")
    except:
        print("Chưa có file csv link")
        return

    all_data = []
    total = len(df)
    for index, row in df.iterrows():
        url = row['url']
        print(f"[{index+1}/{total}] Xử lý: {url}")
        
        data_item = get_detail_content_json(url)
        
        if data_item and data_item['answer']:
            if not data_item['question']:
                data_item['question'] = row['title']
                
            all_data.append(data_item)
    
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=4)
            print(f"Đã lưu {len(all_data)} mục vào JSON.")
            
        time.sleep(random.uniform(1, 2))

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    
    print(f"\nFile kết quả: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()