import json
import re
import os
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from tqdm import tqdm
import unicodedata
from dataclasses import asdict
from bs4 import BeautifulSoup

# Import from existing crawler
try:
    from vbqppl_crawler import HTMLFetcher, ContentExtractor, DocumentContent
except ImportError:
    # Handle case where we run from root
    import sys
    sys.path.append('law-crawler')
    from vbqppl_crawler import HTMLFetcher, ContentExtractor, DocumentContent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qa_crawler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DATASET_PATH = '../data/du_lieu_luat_dataset.json'
MAIN_CORPUS_PATH = '../data/vbqppl_content.json'
OUTPUT_PATH = '../data/qa_vbqppl_content.json'
HTML_DIR = './html_cache_qa'

def normalize_text(text: str) -> str:
    """Normalize text for consistent comparison"""
    if not text:
        return ""
    text = unicodedata.normalize('NFC', text)
    text = text.lower().strip()
    return " ".join(text.split())

class ReferenceParser:
    """Parses reference strings to extract Law Name and Section"""
    
    # Common legal document type prefixes
    DOC_TYPES = [
        "Luật", "Bộ luật", "Nghị quyết", "Nghị định", "Thông tư", 
        "Quyết định", "Pháp lệnh", "Hiến pháp", "Chỉ thị", "Công văn"
    ]
    
    @staticmethod
    def parse(reference: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse reference string.
        Returns: (Law Name, Section Label, Full Reference)
        """
        ref_clean = " ".join(reference.split())
        
        # 1. Try to split by known document types
        # This handles cases like "Khoản 1 Điều 3 Luật Đất đai" -> picks "Luật Đất đai"
        # And "Phần II Nghị định 100/2019" -> picks "Nghị định 100/2019"
        
        best_doc_name = None
        best_section = None
        
        # Find the LAST occurrence of a Doc Type (in case of nested refs, though rare in this dataset)
        # Actually, usually the doc name is at the end.
        
        split_idx = -1
        found_type = ""
        
        for doc_type in ReferenceParser.DOC_TYPES:
            # Look for "Luật", "Nghị định" etc.
            # We want to match whole words to avoiding matching inside other words, 
            # ensuring it's not at the very end (must be followed by something)
            matches = list(re.finditer(fr'\b{doc_type}\b', ref_clean, re.IGNORECASE))
            if matches:
                # Take the last match of this type
                m = matches[-1]
                if m.start() > split_idx:
                    split_idx = m.start()
                    found_type = doc_type

        if split_idx != -1:
            law_name = ref_clean[split_idx:].strip()
            # Cleanup trailing punctuation
            law_name = law_name.strip('.,;"\'')
            
            section_part = ref_clean[:split_idx].strip()
            section_part = section_part.strip('.,;"\'')
            
            # Refine section label: Extract "Điều X" if present in the section part
            # to be more specific, or keep the whole prefix
            if "Điều" in section_part:
                dieu_match = re.search(r'(Điều\s+\d+\w?)', section_part, re.IGNORECASE)
                if dieu_match:
                    section_label = dieu_match.group(1).title()
                else:
                    section_label = section_part
            else:
                section_label = section_part
                
            return law_name, section_label, reference

        # 2. Fallback: If no doc type found, check for "Điều X" and take everything after
        match = re.search(r'(?:^|,\s*)(Điều\s+\d+\w?)(?:[\.,]\s*|\s+)(.+)$', ref_clean, re.IGNORECASE)
        if match:
            section_label = match.group(1).title() 
            law_name = match.group(2).strip()
            law_name = law_name.strip('.,;"\'')
            return law_name, section_label, reference
            
        return None, None, reference

class QAHTMLFetcher(HTMLFetcher):
    """Extended fetcher that handles name-based queries better"""
    
    def extract_document_id(self, vbqppl_name: str) -> str:
        extracted = super().extract_document_id(vbqppl_name)
        if extracted:
            return extracted
        # Fallback: Check for specific patterns like "Luật Đất đai 2024" -> "Luật Đất đai 2024"
        # We return the whole name as ID if no symbol ID found, effectively searching by name
        return vbqppl_name

    def _generate_filename(self, doc_id: str) -> str:
        safe_id = re.sub(r'[\\/*?:"<>|]', '_', doc_id)
        if len(safe_id) > 100:
            safe_id = safe_id[:100]
        hash_suffix = hashlib.md5(doc_id.encode()).hexdigest()[:8]
        return f"{safe_id}_{hash_suffix}.html"

class QAContentExtractor(ContentExtractor):
    """Extended extractor that tries to find the canonical Document ID from HTML"""
    
    @staticmethod
    def extract_canonical_id(html_content: str) -> Optional[str]:
        """
        Attempt to find the official ID (Số hiệu) from the HTML.
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Pattern: Look for "Số hiệu:" in text
            labels = soup.find_all(string=re.compile(r'Số hiệu', re.IGNORECASE))
            for label in labels:
                parent = label.parent
                if parent.name in ['td', 'th']:
                    next_sibling = parent.find_next_sibling(['td', 'th'])
                    if next_sibling:
                        text = next_sibling.get_text(strip=True)
                        if text:
                            return text
            
            # Pattern 2: Specific parsing for LuatVietnam metadata box
            meta_box = soup.select_one('.box-thong-tin, .info-doc')
            if meta_box:
                rows = meta_box.select('tr')
                for row in rows:
                    cells = row.select('td')
                    if len(cells) >= 2:
                        header = cells[0].get_text(strip=True)
                        if 'Số hiệu' in header:
                            return cells[1].get_text(strip=True)
                            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting canonical ID: {e}")
            return None

    @staticmethod
    def extract_from_html(html_content: str, doc_id: str, url: str, 
                          original_name: str = "", original_link: str = "") -> DocumentContent:
        
        # Try to find a better ID
        canonical_id = QAContentExtractor.extract_canonical_id(html_content)
        final_id = canonical_id if canonical_id else doc_id
        
        if canonical_id and canonical_id != doc_id:
            logger.info(f"Resolved canonical ID: {canonical_id} (searched as: {doc_id})")
        
        # Call parent with possibly updated ID
        return ContentExtractor.extract_from_html(
            html_content, final_id, url, original_name, original_link
        )

def main():
    # 1. Load Existing Corpus
    print(f"Loading main corpus from {MAIN_CORPUS_PATH}...")
    existing_docs_map = {}
    
    if os.path.exists(MAIN_CORPUS_PATH):
        try:
            with open(MAIN_CORPUS_PATH, 'r', encoding='utf-8') as f:
                docs = json.load(f)
                for doc in docs:
                    if 'id' in doc:
                        existing_docs_map[normalize_text(doc['id'])] = doc
                        # Also map title/original_name if available for fuzzy lookup
                        if 'title' in doc:
                            existing_docs_map[normalize_text(doc['title'])] = doc
        except Exception as e:
            logger.error(f"Error loading existing corpus: {e}")
            
    print(f"Loaded {len(existing_docs_map)} entries from main corpus.")

    # 2. Load QA Dataset & Identify Required Docs
    print("Loading QA dataset...")
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
    except FileNotFoundError:
         print(f"Dataset not found at {DATASET_PATH}")
         return
    
    required_law_names = set()
    
    for item in qa_data:
        refs = item.get('reference', [])
        for ref in refs:
            law_name, _, _ = ReferenceParser.parse(ref)
            if law_name:
                required_law_names.add(law_name)
    
    print(f"Found {len(required_law_names)} unique document references required.")

    # 3. Separate into Found vs Missing
    qa_corpus_docs: List[Dict] = []
    missing_laws = []
    
    fetcher = QAHTMLFetcher(html_dir=HTML_DIR)

    for law_name in required_law_names:
        # Try to find in existing corpus
        # Strategy 1: strict ID match
        # Strategy 2: name match
        
        doc_id = fetcher.extract_document_id(law_name)
        norm_id = normalize_text(doc_id)
        norm_name = normalize_text(law_name)
        
        found_doc = None
        
        if norm_id in existing_docs_map:
            found_doc = existing_docs_map[norm_id]
        elif norm_name in existing_docs_map:
            found_doc = existing_docs_map[norm_name]
        
        if found_doc:
            # We add a copy to our QA corpus list
            # We don't need to duplicate if we are just outputting a file of RELEVANT docs
            if found_doc not in qa_corpus_docs:
                qa_corpus_docs.append(found_doc)
        else:
            missing_laws.append({'name': law_name})
            
    print(f"Identified {len(missing_laws)} documents to crawl.")
    print(f"Reusing {len(qa_corpus_docs)} documents from existing corpus.")

    # 4. Fetch Missing Documents
    if missing_laws:
        print("Starting crawl for missing documents...")
        results = fetcher.fetch_all(missing_laws)
        
        extractor = QAContentExtractor()
        newly_crawled_count = 0
        
        print("Extracting content for newly crawled documents...")
        for res in results:
            if res.status != 'success' or not res.html_path:
                logger.warning(f"Failed to fetch {res.original_name}: {res.status}")
                continue
                
            try:
                with open(res.html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                guess_id = fetcher.extract_document_id(res.original_name or res.doc_id)
                
                doc_content = extractor.extract_from_html(
                    html_content=html_content,
                    doc_id=guess_id,
                    url=res.url,
                    original_name=res.original_name
                )
                
                if doc_content.status == 'success':
                    doc_dict = asdict(doc_content)
                    qa_corpus_docs.append(doc_dict)
                    newly_crawled_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing {res.doc_id}: {e}")
                
        print(f"Successfully added {newly_crawled_count} new documents.")
    else:
        print("All documents already exist in corpus.")

    # 5. Save QA Corpus
    # Deduplicate by ID just in case
    unique_qa_docs = {}
    for doc in qa_corpus_docs:
        unique_qa_docs[doc['id']] = doc
    
    final_docs_list = list(unique_qa_docs.values())
    
    print(f"Saving {len(final_docs_list)} documents to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_docs_list, f, ensure_ascii=False, indent=4)
    print("Done.")

if __name__ == "__main__":
    main()
