import json
import re
import os
import logging
import difflib
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Logic Helpers ---

def extract_citations_from_text(text: str, strip_small_levels: bool = False, only_article_and_law: bool = False) -> list:
    """Extract structured legal citations. Robust against truncation and missing 'số'."""
    if not text:
        return []
    
    # Normalize text
    text = re.sub(r'\s+', ' ', text)
    
    # Define specialized hierarchical components to avoid false positives like 'điều tra'
    id_val = r"[a-z0-9/.\-–\w]+"
    p_dieu_khoan = r"(?:[Đđ]iều|[Kk]hoản)\s+\d+[a-z0-9/.\-–\w]*"
    p_diem = r"[Đđ]iểm\s+[a-z]\b"
    p_other = r"(?:[Mm]ục|[Cc]hương|[Pp]hần|[Pp]hụ lục)\s+(?:\d+|[ivxlcdm]+)\b"
    
    instruments = r"(?:Nghị quyết|Nghị định|Thông tư|Luật|Bộ luật|Pháp lệnh|Quyết định)"
    law_pattern = f"{instruments}(?:\\s+số)?\\s+[^.\\n,;()<>]{{2,}}"

    hier_pattern = f"(?:{p_dieu_khoan}|{p_diem}|{p_other})"
    # allow 'của ' or ', ' suffix
    part_pattern = f"(?:{hier_pattern}(?:,\\s*|\\s+của\\s+|\\s+)?)?"
    
    # 4. Full Sequence Pattern: parts followed by law context OR just parts
    full_pattern = f"(?:(?:{part_pattern})*{law_pattern}|(?:{part_pattern})+)"
    
    # Find all matches
    matches = re.finditer(full_pattern, text, re.IGNORECASE)
    
    results = []
    for m in matches:
        f = m.group(0).strip().strip(',').strip()
        if len(f) < 5:
            continue
        
        # --- NORMALIZATION LOGIC ---
        if only_article_and_law:
            art_match = re.search(r'[Đđ]iều\s+' + id_val, f, re.IGNORECASE)
            if art_match:
                art = art_match.group(0).strip()
                # Extract the law portion specifically
                law_match = re.search(law_pattern, f, re.IGNORECASE)
                law_context = law_match.group(0).strip() if law_match else ""
                
                if law_context:
                    # Combine Article + Law
                    f = f"{art} {law_context}"
                else:
                    # Fallback to Article only if Law context wasn't found (unlikely)
                    f = art
            
            if strip_small_levels:
                f = re.sub(r'(?:[Đđ]iểm|[Kk]hoản)\s+' + id_val + r'(?:,\s*|\s+của\s+|\s+)?', '', f, flags=re.IGNORECASE)

        elif strip_small_levels:
            f = re.sub(r'(?:[Đđ]iểm|[Kk]hoản)\s+' + id_val + r'(?:,\s*|\s+của\s+|\s+)?', '', f, flags=re.IGNORECASE)
            f = re.sub(r'^\s*của\s+', '', f, flags=re.IGNORECASE)

        f = f.strip().strip(',').strip()
        if len(f) < 5: continue

        # --- POST-EXTRACTION CLEANING (Remove redundant explanation text) ---
        # Stop words that indicate the start of a sentence or explanation
        stop_words = [
            'là', 'theo', 'bởi', 'tại', 'được', 'có', 'khi', 'như', 'về', 
            'của Hội đồng', 'trong trường hợp', 'thì', 'hiện hành', 'quy định',
            'nêu', 'trong', 'của n'
        ]
        stop_pattern = r'\s+(?:' + '|'.join(re.escape(w) for w in stop_words) + r')\b'
        match = re.search(stop_pattern, f, re.IGNORECASE)
        if match:
            f = f[:match.start()].strip()

        # --- FILTER GENERIC & DESCRIPTIVE CITATIONS ---
        f_lower = f.lower()
        instruments_list = ['nghị quyết', 'nghị định', 'thông tư', 'luật', 'bộ luật', 'pháp lệnh', 'quyết định']
        
        # 1. Basic generic check
        placeholders = ['này', 'đó', 'kia', 'hoặc văn bản', 'liên quan', 'nêu trên', 'quy định']
        is_p_generic = f_lower in instruments_list
        if not is_p_generic:
            for inst in instruments_list:
                for p in placeholders:
                    if f_lower == f"{inst} {p}":
                        is_p_generic = True
                        break
                if is_p_generic: break
        if is_p_generic: continue

        # 2. Descriptive check: instrument + lowercase connective (e.g., 'luật của', 'luật và')
        start_inst = None
        for inst in instruments_list:
            if f_lower.startswith(inst):
                start_inst = inst
                break
        
        if start_inst:
            suffix = f[len(start_inst):].strip()
            if suffix:
                first_word = suffix.split()[0]
                # If first word after 'Luật' is lowercase and not 'số', likely descriptive
                if first_word[0].islower() and first_word not in ['số']:
                    continue
        
        # 3. Content-based filter for descriptive legal talk
        descriptive_indicators = [
            'nạn nhân', 'chồng bạn', 'vợ bạn', 'người phạm tội', 'bị kích động',
            'khiến họ', 'dẫn tới', 'lại tiếp diễn', 'làm cho', 'xâm phạm', 'truy cứu'
        ]
        if any(ind in f_lower for ind in descriptive_indicators) and not any(c.isdigit() for c in f):
            continue

        results.append(f)
    
    # Deduplicate
    seen = set()
    cleaned = []
    for f in results:
        f_lower = f.lower()
        if f_lower not in seen:
            seen.add(f_lower)
            cleaned.append(f)
            
    return cleaned

def standardize_citations(citations: list) -> list:
    """Strip small levels, merge standalone articles with context, and deduplicate."""
    if not citations:
        return []

    instruments = r"(?:Nghị quyết|Nghị định|Thông tư|Luật|Bộ luật|Pháp lệnh|Quyết định)"
    
    # 1. Initial stripping and cleaning
    stripped = []
    for c in citations:
        # Standardize: remove redundant words (already handled in extractor, but being safe)
        temp = re.sub(r'(?:[Đđ]iểm|[Kk]hoản)\s+[a-z0-9/.\-–\w]+(?:,\s*|\s+của\s+|\s+)?', '', c, flags=re.IGNORECASE)
        # remove prefix 'của ' if left
        temp = re.sub(r'^\s*của\s+', '', temp, flags=re.IGNORECASE).strip()
        if temp:
            stripped.append(temp)
            
    # 2. Contextual merging (Backwards in legal lists: Điều 1, Điều 2 Luật X -> Điều 1 Luật X, Điều 2 Luật X)
    refined = []
    for i in range(len(stripped)):
        curr = stripped[i]
        
        # Check if curr has an instrument (law root)
        has_law = re.search(instruments, curr, re.IGNORECASE)
        
        if not has_law and i < len(stripped) - 1:
            # Look forward for the first citation with a law context
            next_law_suffix = None
            for j in range(i + 1, len(stripped)):
                match = re.search(instruments, stripped[j], re.IGNORECASE)
                if match:
                    # Extract the law suffix starting from instrument
                    next_law_suffix = stripped[j][match.start():]
                    break
            
            if next_law_suffix:
                if ' của ' not in curr.lower() and not next_law_suffix.lower().startswith('của '):
                    curr = f"{curr} của {next_law_suffix}"
                else:
                    curr = f"{curr} {next_law_suffix}"
        
        refined.append(curr.strip())
        
    # 3. Final case-insensitive deduplication
    seen = set()
    final = []
    for r in refined:
        r_low = r.lower()
        if r_low not in seen:
            seen.add(r_low)
            final.append(r)
    return final

def is_fuzzy_match(s1: str, s2: str, threshold: float = 0.5) -> bool:
    """Check if two strings are a fuzzy match (via sequence similarity or containment)."""
    if not s1 or not s2:
        return False
    # Normalize
    s1, s2 = s1.lower(), s2.lower()
    
    # Strict check first
    if s1 in s2 or s2 in s1:
        return True
        
    # SequenceMatcher for fuzzy
    ratio = difflib.SequenceMatcher(None, s1, s2).ratio()
    return ratio >= threshold

def match_docs_to_citations(gt_citations: list, docs: list, threshold: float = 0.4) -> set:
    """Find which GT citations (Article-level) are covered by hierarchical reconstruction."""
    # 1. Prepare Standardized Article-level GT citations
    # First extract full sequences, then standardize
    clean_gt = standardize_citations(gt_citations)
    
    matched_gt = set()
    
    # 2. Extract citations from docs + reconstruction + standardization
    all_doc_citations = set()
    for doc in docs:
        doc_id = doc.get('doc_id', '')
        title = doc.get('title', '')
        hp = doc.get('hierarchy_path', '')
        
        # Explicitly add raw doc_id to the set of searchable citations
        if doc_id:
            all_doc_citations.add(doc_id.lower())
        
        levels = [l.strip() for l in hp.split('>')] if hp else []
        all_comps = []
        for level in levels:
             all_comps.extend(extract_citations_from_text(level))
        
        # Build reconstruction (Detail -> General)
        detail_to_gen = all_comps[::-1]
        reconstructed = " ".join(detail_to_gen) + f" {title} {doc_id}"
        
        # Extract normalized articles
        doc_cits = extract_citations_from_text(reconstructed.strip(), strip_small_levels=True, only_article_and_law=True)
        # Apply standardization layer to system citations too
        standard_doc_cits = standardize_citations(doc_cits)
        for dc in standard_doc_cits:
            all_doc_citations.add(dc.lower())
            
    # 3. Match
    for gt in clean_gt:
        gt_lower = gt.lower()
        for dc_lower in all_doc_citations:
            if is_fuzzy_match(gt_lower, dc_lower, threshold):
                matched_gt.add(gt.lower())
                break
                
    return matched_gt, clean_gt

def compute_metrics():
    input_file = "../data/evaluation_results.json"
    output_file = "../data/metrics_results.json"
    
    if not os.path.exists(input_file):
        logging.error(f"Input file not found: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle if data is in old format [results...] or new format {"results": [...]}
    if isinstance(data, dict):
        results = data.get("results", [])
    else:
        results = data

    logging.info(f"Processing {len(results)} items...")
    
    all_metrics = []
    overall_retrieval_recalls = []
    overall_retrieval_precisions = []
    overall_citation_recalls = []
    overall_citation_precisions = []

    for item in tqdm(results, desc="Computing Metrics"):
        # 1. Ground Truth Citations (extracted from answer)
        gt_raw = item.get("reference_answer", "")
        gt_original = extract_citations_from_text(gt_raw)
        
        # 2. Retrieval Metrics (Context Docs)
        context_docs = item.get("context_docs", [])
        matched_in_retrieval, clean_gt = match_docs_to_citations(gt_original, context_docs)
        
        r_recall = len(matched_in_retrieval) / len(clean_gt) if clean_gt else 1.0
        
        # Precision for retrieval: (docs that contained at least one relevant stripped citation) / total docs
        relevant_context_count = 0
        for doc in context_docs:
            doc_id = doc.get('doc_id', '')
            title = doc.get('title', '')
            hp = doc.get('hierarchy_path', '')
            
            levels = [l.strip() for l in hp.split('>')] if hp else []
            found_components = []
            for l in levels:
                found_components.extend(extract_citations_from_text(l))
            
            reconstructed = " ".join(found_components[::-1]) + f" {title} {doc_id}"
            doc_citations = extract_citations_from_text(reconstructed, strip_small_levels=True)
            doc_citations_lower = {dc.lower() for dc in doc_citations}
            
            is_relevant = False
            for gt in clean_gt:
                gt_lower = gt.lower()
                if any(is_fuzzy_match(gt_lower, dc_l) for dc_l in doc_citations_lower):
                    is_relevant = True
                    break
            if is_relevant:
                relevant_context_count += 1
                
        r_precision = relevant_context_count / len(context_docs) if context_docs else 0.0

        # 3. Citation Metrics (Used Docs)
        used_docs = item.get("used_docs", [])
        matched_in_citation, _ = match_docs_to_citations(gt_original, used_docs)
        
        c_recall = len(matched_in_citation) / len(clean_gt) if clean_gt else 1.0
        
        # Precision for citation
        relevant_used_count = 0
        for doc in used_docs:
            doc_id = doc.get('doc_id', '')
            title = doc.get('title', '')
            hp = doc.get('hierarchy_path', '')
            
            levels = [l.strip() for l in hp.split('>')] if hp else []
            found_components = []
            for l in levels:
                found_components.extend(extract_citations_from_text(l))
            
            reconstructed = " ".join(found_components[::-1]) + f" {title} {doc_id}"
            doc_citations = extract_citations_from_text(reconstructed, strip_small_levels=True)
            doc_citations_lower = {dc.lower() for dc in doc_citations}
            
            is_relevant = False
            for gt in clean_gt:
                gt_lower = gt.lower()
                if any(is_fuzzy_match(gt_lower, dc_l) for dc_l in doc_citations_lower):
                    is_relevant = True
                    break
            if is_relevant:
                relevant_used_count += 1
                
        c_precision = relevant_used_count / len(used_docs) if used_docs else 0.0

        # Update stats
        if clean_gt: 
            overall_retrieval_recalls.append(r_recall)
            overall_retrieval_precisions.append(r_precision)
            overall_citation_recalls.append(c_recall)
            overall_citation_precisions.append(c_precision)

        all_metrics.append({
            "id": item.get("id"),
            "question": item.get("question"),
            "gt_citations": clean_gt,
            "metrics": {
                "retrieval": {
                    "precision": r_precision,
                    "recall": r_recall,
                    "matched": list(matched_in_retrieval)
                },
                "citation": {
                    "precision": c_precision,
                    "recall": c_recall,
                    "matched": list(matched_in_citation)
                }
            }
        })

    summary = {
        "total_items": len(results),
        "avg_retrieval_recall": sum(overall_retrieval_recalls) / len(overall_retrieval_recalls) if overall_retrieval_recalls else 0,
        "avg_retrieval_precision": sum(overall_retrieval_precisions) / len(overall_retrieval_precisions) if overall_retrieval_precisions else 0,
        "avg_citation_recall": sum(overall_citation_recalls) / len(overall_citation_recalls) if overall_citation_recalls else 0,
        "avg_citation_precision": sum(overall_citation_precisions) / len(overall_citation_precisions) if overall_citation_precisions else 0,
    }

    final_output = {
        "summary": summary,
        "metrics_per_item": all_metrics
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    
    print("\n" + "="*50)
    print("METRICS SUMMARY")
    print("="*50)
    print(f"Total Items: {summary['total_items']}")
    print(f"Retrieval Recall:    {summary['avg_retrieval_recall']:.4f}")
    print(f"Retrieval Precision: {summary['avg_retrieval_precision']:.4f}")
    print(f"Citation Recall:     {summary['avg_citation_recall']:.4f}")
    print(f"Citation Precision:  {summary['avg_citation_precision']:.4f}")
    print("="*50)
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    compute_metrics()
