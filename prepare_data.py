import os
import json
import glob
import subprocess
from docx import Document
from tqdm import tqdm

# Cấu hình
DATA_DIR = "fine_tune_data"
OUTPUT_FILE = "legal_corpus_stage1.jsonl"
MIN_LENGTH = 50  # Bỏ qua các dòng quá ngắn 

def convert_doc_to_docx(doc_path):
    try:
        # Lệnh chạy libreoffice headless để convert
        subprocess.run(
            ['soffice', '--headless', '--convert-to', 'docx', doc_path, '--outdir', DATA_DIR],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        # Trả về đường dẫn file docx mới tạo
        return doc_path + "x" 
    except Exception as e:
        print(f"Lỗi convert file {doc_path}: {e}")
        return None

def read_docx(file_path):
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if len(text) > 0:
                full_text.append(text)
        return "\n".join(full_text)
    except Exception as e:
        print(f"Không đọc được file {file_path}: {e}")
        return ""

def clean_text(text):
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # 1. Loại bỏ dòng quá ngắn (số trang, footer rác)
        # if len(line) < MIN_LENGTH:
        #     continue
            
        # 2. Loại bỏ các dòng header lặp lại quá nhiều (tùy chọn)
        if "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM" in line.upper():
            continue
            
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines)

def main():
    # Lấy danh sách tất cả file trong thư mục data
    all_files = glob.glob(os.path.join(DATA_DIR, "*"))
    
    # Mở file output để ghi (Mode 'w' để ghi mới)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        print(f"Đang xử lý {len(all_files)} file trong thư mục '{DATA_DIR}'")
        
        count = 0
        for file_path in tqdm(all_files):
            filename = os.path.basename(file_path)
            ext = os.path.splitext(filename)[1].lower()
            
            content = ""
            temp_docx = None
            
            # TRƯỜNG HỢP 1: File .docx (Đọc luôn)
            if ext == '.docx':
                content = read_docx(file_path)
                
            # TRƯỜNG HỢP 2: File .doc (Convert -> Đọc -> Xóa tạm)
            elif ext == '.doc':
                temp_docx = convert_doc_to_docx(file_path)
                if temp_docx and os.path.exists(temp_docx):
                    content = read_docx(temp_docx)
                    # Xóa file .docx tạm vừa tạo ra để dọn rác
                    os.remove(temp_docx)
            
            # Bỏ qua các file không phải word
            else:
                continue

            # Xử lý và Ghi dữ liệu
            if content:
                cleaned_content = clean_text(content)
                
                # Chỉ ghi nếu nội dung sau khi clean vẫn còn đủ dài
                if len(cleaned_content) > 200:
                    json_line = json.dumps({"text": cleaned_content}, ensure_ascii=False)
                    f_out.write(json_line + "\n")
                    count += 1

if __name__ == "__main__":
    main()