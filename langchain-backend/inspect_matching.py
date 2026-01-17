import json
import re
from compute_metrics import extract_citations_from_text, standardize_citations, match_docs_to_citations

with open('../data/evaluation_results.json', 'r') as f:
    data = json.load(f)
results = data if isinstance(data, list) else data.get('results', [])

# Show 5 items for deep inspection
for i in range(min(10, len(results))):
    item = results[i]
    gt_raw = item.get('reference_answer', '')
    
    # 1. Extract GT
    gt_original = extract_citations_from_text(gt_raw)
    clean_gt = standardize_citations(gt_original)
    
    # 2. Extract System
    docs = item.get('used_docs', [])
    matched, _ = match_docs_to_citations(gt_original, docs, threshold=0.51)
    
    sys_citations = []
    for doc in docs:
        doc_id = doc.get('doc_id', '')
        title = doc.get('title', '')
        hp = doc.get('hierarchy_path', '')
        levels = [l.strip() for l in hp.split('>')] if hp else []
        all_comps = []
        for level in levels:
             all_comps.extend(extract_citations_from_text(level))
        reconstructed = ' '.join(all_comps[::-1]) + f' {title} {doc_id}'
        found = extract_citations_from_text(reconstructed.strip(), only_article_and_law=True, strip_small_levels=True)
        sys_citations.extend(standardize_citations(found))
    
    print(f'--- Item {i} ---')
    print(f'Question: {item.get("question")[:80]}...')
    print(f'Clean GT: {clean_gt}')
    print(f'System Extracted: {list(set(sys_citations))}')
    print(f'Matched: {list(matched)}\n')