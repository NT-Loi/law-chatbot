"""
VBQPPL Crawler for luatvietnam.vn

Two-phase crawling strategy:
1. Phase 1 (Multithreading): Fetch all HTML content and save to disk
2. Phase 2 (Multiprocessing): Extract content from saved HTML files

This approach optimizes for:
- I/O-bound operations (fetching) using ThreadPoolExecutor
- CPU-bound operations (parsing) using ProcessPoolExecutor
"""

import json
import re
import time
import os
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from urllib.parse import urljoin, quote
import multiprocessing

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s',
    handlers=[
        logging.FileHandler('vbqppl_crawler.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Section:
    """Represents a document section (Điều) with hierarchical context"""
    label: str  # e.g., "Điều 1. Phạm vi điều chỉnh"
    content: str  # The content of this section
    hierarchy_path: str  # e.g., "Chương I QUY ĐỊNH CHUNG > Điều 1. Phạm vi điều chỉnh"


@dataclass
class DocumentContent:
    """Represents extracted document content"""
    id: str
    title: str
    url: str
    content: str
    status: str  # 'success', 'pdf_skip', 'not_found', 'error'
    error_message: Optional[str] = None
    original_name: Optional[str] = None
    original_link: Optional[str] = None
    sections: Optional[List[Dict[str, str]]] = None  # List of sections with hierarchy paths


@dataclass 
class FetchResult:
    """Result of fetching HTML from a URL"""
    doc_id: str
    url: str
    html_path: Optional[str]  # Path to saved HTML file
    status: str  # 'success', 'not_found', 'error'
    error_message: Optional[str] = None
    original_name: Optional[str] = None
    original_link: Optional[str] = None


class HTMLFetcher:
    """Handles fetching and saving HTML content using multithreading"""
    
    BASE_URL = "https://luatvietnam.vn"
    SEARCH_URL = "https://luatvietnam.vn/van-ban/tim-van-ban.html"
    
    def __init__(self, html_dir: str, max_workers: int = 10, delay: float = 0.5):
        """
        Initialize the HTML fetcher
        
        Args:
            html_dir: Directory to save HTML files
            max_workers: Maximum number of concurrent threads
            delay: Delay between requests per thread
        """
        self.html_dir = Path(html_dir)
        self.html_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.delay = delay
        
        # Create a session for each thread
        self._local = None
    
    def _get_session(self) -> requests.Session:
        """Get or create a thread-local session"""
        import threading
        if not hasattr(self, '_sessions'):
            self._sessions = {}
        
        thread_id = threading.current_thread().ident
        if thread_id not in self._sessions:
            session = requests.Session()
            
            # Increase connection pool size to handle concurrent threads (avoiding warnings)
            from requests.adapters import HTTPAdapter
            adapter = HTTPAdapter(pool_connections=20, pool_maxsize=30)
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'vi,en-US;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Referer': 'https://luatvietnam.vn/',
            })
            self._sessions[thread_id] = session
        
        return self._sessions[thread_id]
    
    def extract_document_id(self, vbqppl_name: str) -> Optional[str]:
        """Extract document ID from VBQPPL name"""
        patterns = [
            # Multi-part IDs with hyphens: e.g., 127/2007/TTLT-BQP-CCBVN
            r'(\d+/\d{4}/[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ]+(?:-[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ0-9]+)+)',
            # IDs with suffix numbers: e.g., 13/2018/QH14, 65/2014/QH13
            r'(\d+/\d{4}/[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ]+\d+)',
            # Simple IDs: e.g., 15/2020/NĐ-CP
            r'(\d+/\d{4}/[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ]+-[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ]+)',
            # Fallback: Simple alphanumeric pattern
            r'(\d+/\d{4}/[A-Z]+(?:-[A-Z0-9]+)*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, vbqppl_name, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _generate_filename(self, doc_id: str) -> str:
        """Generate a safe filename from document ID"""
        # Replace / with _ and create hash for uniqueness
        safe_id = doc_id.replace('/', '_').replace('-', '_')
        hash_suffix = hashlib.md5(doc_id.encode()).hexdigest()[:8]
        return f"{safe_id}_{hash_suffix}.html"
    
    def search_document(self, doc_id: str) -> Optional[str]:
        """Search for a document and return its URL"""
        session = self._get_session()
        
        try:
            search_params = {'Keywords': doc_id}
            response = session.get(self.SEARCH_URL, params=search_params, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the first search result link
            result_links = soup.select('.doc-title a')
            if not result_links:
                result_links = soup.select('.art-search a[href*="-d1.html"], a[href*="-d1.html"]')
            
            for link in result_links:
                href = link.get('href', '')
                if href and '-d1.html' in href:
                    return urljoin(self.BASE_URL, href) if not href.startswith('http') else href
            
            return None
            
        except Exception as e:
            logger.error(f"Error searching for document {doc_id}: {e}")
            return None
    
    def fetch_and_save(self, vbqppl: Dict[str, Any]) -> FetchResult:
        """
        Fetch HTML for a single VBQPPL entry and save to disk
        
        Args:
            vbqppl: VBQPPL dictionary with 'name' and optional 'link' fields
            
        Returns:
            FetchResult with status and path to saved HTML
        """
        name = vbqppl.get('name', '')
        original_link = vbqppl.get('link', '')
        
        # Extract document ID
        doc_id = self.extract_document_id(name)
        if not doc_id:
            return FetchResult(
                doc_id="",
                url="",
                html_path=None,
                status="error",
                error_message=f"Could not extract document ID from: {name[:50]}",
                original_name=name,
                original_link=original_link
            )
        
        # Check if already fetched
        filename = self._generate_filename(doc_id)
        html_path = self.html_dir / filename
        if html_path.exists():
            logger.info(f"Already fetched: {doc_id}")
            return FetchResult(
                doc_id=doc_id,
                url="",
                html_path=str(html_path),
                status="cached",
                original_name=name,
                original_link=original_link
            )
        
        # Add delay to be respectful
        time.sleep(self.delay)
        
        # Search for document URL
        logger.info(f"Searching for: {doc_id}")
        doc_url = self.search_document(doc_id)
        
        if not doc_url:
            return FetchResult(
                doc_id=doc_id,
                url="",
                html_path=None,
                status="not_found",
                error_message="Document not found in search results",
                original_name=name,
                original_link=original_link
            )
        
        # Fetch the document HTML
        try:
            time.sleep(self.delay)  # Additional delay before fetching
            session = self._get_session()
            response = session.get(doc_url, timeout=30)
            response.raise_for_status()
            
            # Save HTML to disk
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            logger.info(f"Saved HTML: {doc_id} -> {filename}")
            return FetchResult(
                doc_id=doc_id,
                url=doc_url,
                html_path=str(html_path),
                status="success",
                original_name=name,
                original_link=original_link
            )
            
        except Exception as e:
            logger.error(f"Error fetching {doc_id}: {e}")
            return FetchResult(
                doc_id=doc_id,
                url=doc_url,
                html_path=None,
                status="error",
                error_message=str(e),
                original_name=name,
                original_link=original_link
            )
    
    def fetch_all(self, vbqppl_list: List[Dict[str, Any]], 
                  progress_callback: Optional[callable] = None) -> List[FetchResult]:
        """
        Fetch all VBQPPL documents using multithreading
        
        Args:
            vbqppl_list: List of VBQPPL dictionaries
            progress_callback: Optional callback function(completed, total)
            
        Returns:
            List of FetchResult objects
        """
        results = []
        
        # Deduplicate based on extracted document ID
        unique_vbqppl = {}
        logger.info("Extracting IDs and deduplicating...")
        
        for vbqppl in vbqppl_list:
            name = vbqppl.get('name', '')
            doc_id = self.extract_document_id(name)
            
            if doc_id and doc_id not in unique_vbqppl:
                unique_vbqppl[doc_id] = vbqppl
        
        total = len(unique_vbqppl)
        logger.info(f"Refined list: {total} unique documents to fetch (from {len(vbqppl_list)} total entries)")
        logger.info(f"Starting to fetch with {self.max_workers} threads")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_vbqppl = {
                executor.submit(self.fetch_and_save, vbqppl): vbqppl 
                for vbqppl in unique_vbqppl.values()
            }
            
            # Process completed tasks with tqdm progress bar
            with tqdm(total=total, desc="📥 Fetching HTML", unit="doc", 
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                for future in as_completed(future_to_vbqppl):
                    try:
                        result = future.result()
                        results.append(result)
                        # Update progress bar postfix with current doc
                        pbar.set_postfix_str(f"{result.doc_id[:20]}..." if len(result.doc_id) > 20 else result.doc_id)
                    except Exception as e:
                        vbqppl = future_to_vbqppl[future]
                        logger.error(f"Task failed for {vbqppl.get('name', '')[:30]}: {e}")
                        results.append(FetchResult(
                            doc_id="",
                            url="",
                            html_path=None,
                            status="error",
                            error_message=str(e),
                            original_name=vbqppl.get('name', ''),
                            original_link=vbqppl.get('link', '')
                        ))
                    
                    pbar.update(1)
                    if progress_callback:
                        progress_callback(pbar.n, total)
        
        return results


class ContentExtractor:
    """Handles extracting content from saved HTML files using multiprocessing"""
    
    @staticmethod
    def table_to_markdown(table: BeautifulSoup) -> str:
        """Convert an HTML table to markdown format"""
        rows = []
        
        for tr in table.find_all('tr'):
            cells = []
            for cell in tr.find_all(['th', 'td']):
                text = cell.get_text(separator=' ', strip=True)
                text = text.replace('|', '\\|')
                text = ' '.join(text.split())
                cells.append(text)
            
            if cells:
                rows.append(cells)
        
        if not rows:
            return ""
        
        markdown_lines = []
        max_cols = max(len(row) for row in rows)
        
        for i, row in enumerate(rows):
            while len(row) < max_cols:
                row.append('')
        
        if rows:
            markdown_lines.append('| ' + ' | '.join(rows[0]) + ' |')
            markdown_lines.append('| ' + ' | '.join(['---'] * max_cols) + ' |')
            
            for row in rows[1:]:
                markdown_lines.append('| ' + ' | '.join(row) + ' |')
        
        return '\n'.join(markdown_lines)
    
    # Regex patterns for structural headers
    STRUCTURE_PATTERNS = {
        'phan': re.compile(r'^(Phần|PHẦN)\s+([IVXLCDM]+|\d+)', re.IGNORECASE),
        'chuong': re.compile(r'^(Chương|CHƯƠNG)\s+([IVXLCDM]+|\d+)', re.IGNORECASE),
        'muc': re.compile(r'^(Mục|MỤC)\s+(\d+)', re.IGNORECASE),
        # Điều must be followed by number and then period or colon (not part of a longer phrase)
        'dieu': re.compile(r'^(Điều|ĐIỀU)\s+(\d+)[\.\:]', re.IGNORECASE),
        'phu_luc': re.compile(r'^(Phụ lục|PHỤ LỤC)\s*([IVXLCDM]+|\d+)?', re.IGNORECASE),
    }
    
    @staticmethod
    def _extract_document_title(content_elem: BeautifulSoup) -> str:
        """Extract the document title from content (usually docitem-13 class, bold/centered)"""
        # Try docitem-13 which is typically the document title
        title_elem = content_elem.select_one('.docitem-13')
        if title_elem:
            # Get clean text, removing any inner hidden elements first
            for hidden in title_elem.select('.target-hidden, .bg-theo-doi, .tooltip-button'):
                hidden.decompose()
            title = title_elem.get_text(separator=' ', strip=True)
            # Clean up underscores that are sometimes used as separators
            title = re.sub(r'_{2,}', '', title)
            # Normalize whitespace
            title = re.sub(r'\s+', ' ', title)
            return title.strip()
        return ""
    
    @staticmethod
    def _get_section_text(elem: BeautifulSoup) -> str:
        """Get clean text from an element, handling tables and deduplication"""
        # Clone the element to avoid modifying original
        from copy import copy
        elem_copy = copy(elem)
        
        # Remove hidden elements
        for hidden in elem_copy.select('.target-hidden, .bg-theo-doi, .tooltip-button'):
            hidden.decompose()
        
        # Get text while preserving table structure
        text_parts = []
        for child in elem_copy.children:
            if hasattr(child, 'name'):
                if child.name == 'table':
                    text_parts.append('\n' + ContentExtractor.table_to_markdown(child) + '\n')
                else:
                    text_parts.append(child.get_text(strip=True))
            elif isinstance(child, str) and child.strip():
                text_parts.append(child.strip())
        
        return ' '.join(text_parts)
    
    @staticmethod
    def _identify_header_type(text: str) -> Optional[str]:
        """Identify if text is a structural header and return its type"""
        for header_type, pattern in ContentExtractor.STRUCTURE_PATTERNS.items():
            if pattern.match(text):
                return header_type
        return None
    
    @staticmethod
    def _clean_header_text(text: str) -> str:
        """Clean header text and format properly with spaces"""
        # Get just the first line
        lines = text.split('\n')
        header_text = lines[0].strip() if lines else text
        
        # Fix missing space after Roman numerals/numbers in headers
        # Simple approach: insert space between Roman numeral chars and Vietnamese chars
        header_text = re.sub(
            r'^(Chương|CHƯƠNG|Mục|MỤC|Phần|PHẦN|Phụ lục|PHỤ LỤC)\s*([IVXLCDM]+|[0-9]+)\s*([A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ])',
            r'\1 \2 \3',
            header_text
        )
        
        # Also fix missing space in "Điều 1.Phạm vi" -> "Điều 1. Phạm vi"
        header_text = re.sub(r'(Điều|ĐIỀU)\s*(\d+)\.([A-ZĐa-zđ])', r'\1 \2. \3', header_text)
        
        # Normalize multiple spaces
        header_text = re.sub(r'\s+', ' ', header_text)
        
        return header_text.strip()
    
    @staticmethod
    def extract_from_html(html_content: str, doc_id: str, url: str, 
                          original_name: str = "", original_link: str = "") -> DocumentContent:
        """
        Extract document content from HTML string with hierarchical structure
        
        Args:
            html_content: Raw HTML content
            doc_id: Document ID
            url: Original document URL
            original_name: Original VBQPPL name
            original_link: Original VBQPPL link
            
        Returns:
            DocumentContent object with sections containing hierarchy paths
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Get page title as fallback
            page_title_elem = soup.select_one('h1.title, .document-title, .vb-title, .title-detail h1, .detail-title')
            page_title = page_title_elem.get_text(strip=True) if page_title_elem else "Unknown Title"
            
            # Check for PDF/updating content
            noidung_elem = soup.select_one('#noidung')
            if noidung_elem:
                noidung_text = noidung_elem.get_text()
                if 'đang được cập nhật' in noidung_text.lower() and len(noidung_text) < 200:
                    return DocumentContent(
                        id=doc_id,
                        title=page_title,
                        url=url,
                        content="",
                        status="pdf_skip",
                        error_message="Document content is only available as PDF or being updated",
                        original_name=original_name,
                        original_link=original_link
                    )
            
            # Check for PDF indicators
            pdf_indicators = [
                soup.select_one('.pdf-only'),
                soup.select_one('[data-content-type="pdf"]'),
                soup.select_one('iframe[src*=".pdf"]'),
                soup.select_one('.content-pdf'),
            ]
            
            if any(indicator for indicator in pdf_indicators):
                return DocumentContent(
                    id=doc_id,
                    title=page_title,
                    url=url,
                    content="",
                    status="pdf_skip",
                    error_message="Document content is only available as PDF",
                    original_name=original_name,
                    original_link=original_link
                )
            
            # Find content container
            content_selectors = [
                '#noidung .the-document-body',
                '#noidung .the-document-entry',
                '#noidung',
                '.the-document-body',
                '.content-vb',
                '.main-content',
            ]
            
            content_elem = None
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    text = content_elem.get_text(strip=True)
                    if len(text) > 100:
                        break
                    content_elem = None
            
            if not content_elem:
                return DocumentContent(
                    id=doc_id,
                    title=page_title,
                    url=url,
                    content="",
                    status="error",
                    error_message="Could not find content container",
                    original_name=original_name,
                    original_link=original_link
                )
            
            # Remove hidden elements and tooltips BEFORE any processing
            for hidden in content_elem.select('.target-hidden, .bg-theo-doi, .tooltip-button'):
                hidden.decompose()
            
            # Extract document title from content (docitem-13)
            doc_title = ContentExtractor._extract_document_title(content_elem)
            if not doc_title:
                doc_title = page_title
            
            # ============================================
            # HIERARCHICAL STRUCTURE EXTRACTION
            # ============================================
            # Track current hierarchy: Phần > Chương > Mục > Điều/Phụ lục
            hierarchy = {
                'phan': None,
                'chuong': None,
                'muc': None,
            }
            
            sections = []  # List of extracted sections with hierarchy paths
            current_section_content = []
            current_section_label = None
            current_section_type = None
            
            def _build_hierarchy_path():
                """Build the hierarchy path string"""
                parts = []
                if hierarchy['phan']:
                    parts.append(hierarchy['phan'])
                if hierarchy['chuong']:
                    parts.append(hierarchy['chuong'])
                if hierarchy['muc']:
                    parts.append(hierarchy['muc'])
                return ' > '.join(parts) if parts else ""
            
            def _save_current_section():
                """Save the current section to sections list"""
                nonlocal current_section_content, current_section_label, current_section_type
                if current_section_label and current_section_content:
                    content_text = '\n'.join(current_section_content).strip()
                    content_text = re.sub(r'\n{3,}', '\n\n', content_text)
                    if content_text:
                        hierarchy_path = _build_hierarchy_path()
                        if hierarchy_path:
                            full_path = f"{hierarchy_path} > {current_section_label}"
                        else:
                            full_path = current_section_label
                        
                        sections.append({
                            'label': current_section_label,
                            'content': content_text,
                            'hierarchy_path': full_path,
                            'type': current_section_type or 'unknown'
                        })
                current_section_content = []
                current_section_label = None
                current_section_type = None
            
            # Process structural elements
            # Look for docitem-2 (Chương headers), docitem-5 (Điều headers), etc.
            # Include 'table' in search to handle markdown conversion
            all_elements = content_elem.find_all(['p', 'div', 'span', 'table'], recursive=True)
            processed_headers = set()  # Track processed headers to avoid duplicates
            
            for elem in all_elements:
                classes = elem.get('class', [])
                
                # Special handling for tables
                if elem.name == 'table':
                    # Skip nested tables (handled by outer table)
                    if elem.find_parent('table'):
                        continue
                    text = ContentExtractor.table_to_markdown(elem)
                    header_type = None # Tables are content, not headers
                else:
                    # For non-table elements
                    # If this element strictly contains a table, skip it and let the table be processed separately
                    # This avoids flattening the table into text
                    if elem.find('table'):
                        continue
                        
                    # Skip content that is inside a table (already handled by table_to_markdown)
                    if elem.find_parent('table'):
                        continue
                        
                    text = elem.get_text(strip=True)
                
                if not text or len(text) < 3:
                    continue
                
                # Skip duplicate content from nested elements
                # Check if this element's text is a subset of parent's text
                parent = elem.parent
                if parent and parent != content_elem and parent.name in ['p', 'div', 'span']:
                    # Only check for duplicate if parent was NOT skipped
                    # Parent is skipped if it contains a table
                    parent_has_table = parent.find('table')
                    
                    if not parent_has_table:
                        parent_text = parent.get_text(strip=True)
                        # Skip if this element's text is contained in parent
                        # This covers both substring and exact equality
                        if text in parent_text:
                            continue
                
                # Identify header type
                header_type = ContentExtractor._identify_header_type(text)
                
                if header_type:
                    clean_text = ContentExtractor._clean_header_text(text)
                    
                    if header_type == 'phan':
                        # New Phần resets everything below it
                        _save_current_section()
                        hierarchy['phan'] = clean_text
                        hierarchy['chuong'] = None
                        hierarchy['muc'] = None
                        
                    elif header_type == 'chuong':
                        # New Chương resets Mục
                        _save_current_section()
                        hierarchy['chuong'] = clean_text
                        hierarchy['muc'] = None
                        
                    elif header_type == 'muc':
                        _save_current_section()
                        hierarchy['muc'] = clean_text
                        
                    elif header_type in ('dieu', 'phu_luc'):
                        # Start a new section for Điều or Phụ lục
                        _save_current_section()
                        current_section_label = clean_text
                        current_section_type = header_type
                        
                        # Try to get the content following this header
                        # The content is typically in subsequent sibling elements or child mab2 elements
                        mab2_elem = elem.select_one('.mab2')
                        if mab2_elem:
                            # Content is inside this element after the label
                            full_text = elem.get_text(strip=True)
                            # Remove the label part to get content
                            content_after_label = full_text[len(clean_text):].strip()
                            if content_after_label:
                                current_section_content.append(content_after_label)
                        
                elif current_section_label:
                    # We're inside a section (after a Điều/Phụ lục), collect content
                    # Check if this is a content element (docitem-11, docitem-12, etc)
                    is_content_class = any(c.startswith('docitem-') and c not in ['docitem-2', 'docitem-5', 'docitem-13', 'docitem-14'] 
                                          for c in classes)
                    
                    # Also check for mab2 which contains clean text
                    if 'mab2' in classes or is_content_class or not classes:
                        # Avoid adding header text again
                        if not ContentExtractor._identify_header_type(text):
                            current_section_content.append(text)
            
            # Save the last section
            _save_current_section()
            
            # ============================================
            # FALLBACK: Full content extraction
            # ============================================
            content_parts = []
            seen_texts = set()
            processed_tables = set()
            
            # Process tables
            for table in content_elem.find_all('table'):
                table_id = id(table)
                if table_id not in processed_tables:
                    processed_tables.add(table_id)
                    md_table = ContentExtractor.table_to_markdown(table)
                    if md_table and md_table not in seen_texts:
                        content_parts.append('\n\n' + md_table + '\n\n')
                        seen_texts.add(md_table)
                        seen_texts.add(table.get_text(strip=True))
            
            # Process text elements
            for element in content_elem.find_all(['p', 'div', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'b', 'strong', 'i', 'em']):
                if element.find_parent('table'):
                    continue
                
                parent = element.parent
                if parent and parent.name in ['p', 'div', 'span'] and parent.get_text(strip=True) == element.get_text(strip=True):
                    continue
                
                text = element.get_text(strip=True)
                if text and len(text) > 5 and text not in seen_texts:
                    is_duplicate = any(text in seen for seen in seen_texts if len(seen) > len(text))
                    if not is_duplicate:
                        content_parts.append(text + '\n')
                        seen_texts.add(text)
            
            if not content_parts:
                raw_content = content_elem.get_text(separator='\n', strip=True)
                content_parts = [raw_content]
            
            full_content = '\n'.join(content_parts)
            full_content = re.sub(r'\n{3,}', '\n\n', full_content)
            full_content = full_content.strip()
            
            if not full_content and not sections:
                return DocumentContent(
                    id=doc_id,
                    title=doc_title,
                    url=url,
                    content="",
                    status="error",
                    error_message="Content is empty",
                    original_name=original_name,
                    original_link=original_link
                )
            
            return DocumentContent(
                id=doc_id,
                title=doc_title,
                url=url,
                content=full_content,
                status="success",
                original_name=original_name,
                original_link=original_link,
                sections=sections if sections else None
            )
            
        except Exception as e:
            return DocumentContent(
                id=doc_id,
                title="Unknown",
                url=url,
                content="",
                status="error",
                error_message=str(e),
                original_name=original_name,
                original_link=original_link
            )
    
    @staticmethod
    def extract_from_file(args: Tuple[str, str, str, str, str]) -> DocumentContent:
        """
        Extract content from a saved HTML file (for multiprocessing)
        
        Args:
            args: Tuple of (html_path, doc_id, url, original_name, original_link)
            
        Returns:
            DocumentContent object
        """
        html_path, doc_id, url, original_name, original_link = args
        
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            return ContentExtractor.extract_from_html(
                html_content, doc_id, url, original_name, original_link
            )
        except Exception as e:
            return DocumentContent(
                id=doc_id,
                title="Unknown",
                url=url,
                content="",
                status="error",
                error_message=f"Error reading HTML file: {e}",
                original_name=original_name,
                original_link=original_link
            )


class VBQPPLCrawler:
    """Main crawler class orchestrating the two-phase crawling process"""
    
    def __init__(self, html_dir: str = "./html_cache", 
                 fetch_workers: int = 10,
                 extract_workers: int = None,
                 fetch_delay: float = 0.5):
        """
        Initialize the crawler
        
        Args:
            html_dir: Directory to save fetched HTML files
            fetch_workers: Number of threads for fetching (Phase 1)
            extract_workers: Number of processes for extraction (Phase 2), defaults to CPU count
            fetch_delay: Delay between requests per thread
        """
        self.html_dir = html_dir
        self.fetcher = HTMLFetcher(html_dir, max_workers=fetch_workers, delay=fetch_delay)
        self.extract_workers = extract_workers or max(1, multiprocessing.cpu_count() - 4)
    
    def phase1_fetch(self, vbqppl_list: List[Dict[str, Any]], 
                     checkpoint_file: str = None) -> List[FetchResult]:
        """
        Phase 1: Fetch all HTML content using multithreading
        
        Args:
            vbqppl_list: List of VBQPPL dictionaries
            checkpoint_file: File to save fetch results for checkpointing
            
        Returns:
            List of FetchResult objects
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: Fetching HTML content with multithreading")
        logger.info(f"Total documents to fetch: {len(vbqppl_list)}")
        logger.info(f"Using {self.fetcher.max_workers} threads")
        logger.info("=" * 60)
        
        start_time = time.time()
        results = self.fetcher.fetch_all(vbqppl_list)
        elapsed = time.time() - start_time
        
        # Statistics
        success = sum(1 for r in results if r.status in ['success', 'cached'])
        not_found = sum(1 for r in results if r.status == 'not_found')
        errors = sum(1 for r in results if r.status == 'error')
        
        logger.info(f"Phase 1 completed in {elapsed:.2f}s")
        logger.info(f"Success: {success}, Not Found: {not_found}, Errors: {errors}")
        
        # Save checkpoint
        if checkpoint_file:
            checkpoint_data = [asdict(r) for r in results]
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved fetch checkpoint to: {checkpoint_file}")
        
        return results
    
    def phase2_extract(self, fetch_results: List[FetchResult],
                       output_file: str = None) -> List[DocumentContent]:
        """
        Phase 2: Extract content from saved HTML files using multiprocessing
        
        Args:
            fetch_results: List of FetchResult objects from Phase 1
            output_file: File to save extracted content
            
        Returns:
            List of DocumentContent objects
        """
        # Filter only successful fetches
        valid_results = [r for r in fetch_results if r.html_path and r.status in ['success', 'cached']]
        
        logger.info("=" * 60)
        logger.info("PHASE 2: Extracting content with multiprocessing")
        logger.info(f"Documents to extract: {len(valid_results)}")
        logger.info(f"Using {self.extract_workers} processes")
        logger.info("=" * 60)
        
        if not valid_results:
            logger.warning("No valid documents to extract")
            return []
        
        # Prepare arguments for multiprocessing
        extract_args = [
            (r.html_path, r.doc_id, r.url, r.original_name or "", r.original_link or "")
            for r in valid_results
        ]
        
        start_time = time.time()
        extracted = []
        
        # Use multiprocessing for CPU-bound extraction
        with ProcessPoolExecutor(max_workers=self.extract_workers) as executor:
            futures = {executor.submit(ContentExtractor.extract_from_file, args): args[1] 
                       for args in extract_args}
            
            with tqdm(total=len(valid_results), desc="📄 Extracting Content", unit="doc",
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                for future in as_completed(futures):
                    doc_id = futures[future]
                    try:
                        result = future.result()
                        extracted.append(result)
                    except Exception as e:
                        logger.error(f"Extraction failed for {doc_id}: {e}")
                        extracted.append(DocumentContent(
                            id=doc_id,
                            title="Unknown",
                            url="",
                            content="",
                            status="error",
                            error_message=str(e)
                        ))
                    
                    pbar.set_postfix_str(f"{doc_id[:20]}..." if len(doc_id) > 20 else doc_id)
                    pbar.update(1)
        
        elapsed = time.time() - start_time
        
        # Statistics
        success = sum(1 for r in extracted if r.status == 'success')
        pdf_skip = sum(1 for r in extracted if r.status == 'pdf_skip')
        errors = sum(1 for r in extracted if r.status == 'error')
        
        logger.info(f"Phase 2 completed in {elapsed:.2f}s")
        logger.info(f"Success: {success}, PDF Skip: {pdf_skip}, Errors: {errors}")
        
        # Also add not_found results from phase 1
        for r in fetch_results:
            if r.status == 'not_found':
                extracted.append(DocumentContent(
                    id=r.doc_id,
                    title="",
                    url="",
                    content="",
                    status="not_found",
                    error_message="Document not found in search results",
                    original_name=r.original_name,
                    original_link=r.original_link
                ))
        
        # Save results
        if output_file:
            output_data = [asdict(r) for r in extracted]
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved extracted content to: {output_file}")
        
        return extracted
    
    def run(self, vbqppl_list: List[Dict[str, Any]], 
            output_file: str,
            checkpoint_file: str = None) -> List[DocumentContent]:
        """
        Run the complete two-phase crawling process
        
        Args:
            vbqppl_list: List of VBQPPL dictionaries
            output_file: File to save final extracted content
            checkpoint_file: File to save/load fetch checkpoint
            
        Returns:
            List of DocumentContent objects
        """
        # Check for existing checkpoint
        fetch_results = None
        if checkpoint_file and os.path.exists(checkpoint_file):
            logger.info(f"Loading fetch checkpoint from: {checkpoint_file}")
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            fetch_results = [FetchResult(**r) for r in checkpoint_data]
            logger.info(f"Loaded {len(fetch_results)} fetch results from checkpoint")
        
        # Phase 1: Fetch HTML
        if fetch_results is None:
            fetch_results = self.phase1_fetch(vbqppl_list, checkpoint_file)
        
        # Phase 2: Extract content
        extracted = self.phase2_extract(fetch_results, output_file)
        
        return extracted


def load_dieu_json(file_path: str) -> List[Dict[str, Any]]:
    """Load and parse Dieu.json file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_all_vbqppl(dieu_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract all unique VBQPPL entries from Dieu list"""
    vbqppl_set = set()
    vbqppl_list = []
    
    for dieu in dieu_list:
        for vbqppl in dieu.get('VBQPPL', []):
            name = vbqppl.get('name', '')
            if name and name not in vbqppl_set:
                vbqppl_set.add(name)
                vbqppl_list.append(vbqppl)
    
    return vbqppl_list


def main():
    """Main function to run the crawler"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Crawl VBQPPL documents from luatvietnam.vn (Two-phase strategy)')
    parser.add_argument('--input', '-i', type=str, default='../data/phap_dien/Dieu.json',
                        help='Path to Dieu.json file')
    parser.add_argument('--output', '-o', type=str, default='../data/vbqppl_content.json',
                        help='Path to output JSON file')
    parser.add_argument('--html-dir', type=str, default='./html_cache',
                        help='Directory to cache fetched HTML files')
    parser.add_argument('--checkpoint', '-c', type=str, default='./fetch_checkpoint.json',
                        help='Path to fetch checkpoint file')
    parser.add_argument('--fetch-workers', type=int, default=20,
                        help='Number of threads for fetching (Phase 1)')
    parser.add_argument('--extract-workers', type=int, default=12,
                        help='Number of processes for extraction (Phase 2), defaults to CPU count - 4')
    parser.add_argument('--delay', '-d', type=float, default=0.5,
                        help='Delay between requests per thread in seconds')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Limit number of documents to process')
    parser.add_argument('--phase', type=str, choices=['all', '1', '2'], default='all',
                        help='Which phase to run: all, 1 (fetch only), 2 (extract only)')
    
    args = parser.parse_args()
    
    # Initialize crawler
    crawler = VBQPPLCrawler(
        html_dir=args.html_dir,
        fetch_workers=args.fetch_workers,
        extract_workers=args.extract_workers,
        fetch_delay=args.delay
    )
    
    if args.phase == '2':
        # Phase 2 only - load checkpoint and extract
        if not os.path.exists(args.checkpoint):
            logger.error(f"Checkpoint file not found: {args.checkpoint}")
            return
        
        with open(args.checkpoint, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        fetch_results = [FetchResult(**r) for r in checkpoint_data]
        
        crawler.phase2_extract(fetch_results, args.output)
    else:
        # Load Dieu.json
        logger.info(f"Loading input file: {args.input}")
        dieu_list = load_dieu_json(args.input)
        logger.info(f"Loaded {len(dieu_list)} Dieu entries")
        
        # Extract all unique VBQPPL entries
        vbqppl_list = extract_all_vbqppl(dieu_list)
        logger.info(f"Found {len(vbqppl_list)} unique VBQPPL entries")
        
        # Apply limit if specified
        if args.limit:
            vbqppl_list = vbqppl_list[:args.limit]
            logger.info(f"Limited to {args.limit} entries")
        
        if args.phase == '1':
            # Phase 1 only - fetch and save checkpoint
            crawler.phase1_fetch(vbqppl_list, args.checkpoint)
        else:
            # Run both phases
            crawler.run(vbqppl_list, args.output, args.checkpoint)
    
    logger.info("Done!")


if __name__ == '__main__':
    main()