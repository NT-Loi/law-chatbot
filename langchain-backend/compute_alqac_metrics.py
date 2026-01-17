import json
import os
import argparse
from collections import Counter

def compute_metrics(results_path):
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        return

    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    if not results:
        print("Error: Results file is empty.")
        return

    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    
    # Answer Accuracy
    types = sorted(list(set(r['type'] for r in results)))
    accuracy_by_type = {}
    for t in types:
        type_results = [r for r in results if r['type'] == t]
        type_total = len(type_results)
        type_correct = sum(1 for r in type_results if r['is_correct'])
        accuracy_by_type[t] = {
            'total': type_total,
            'correct': type_correct,
            'accuracy': type_correct / type_total if type_total > 0 else 0
        }

    # Retrieval & Citation Metrics
    # Filter for items that actually have ground truth
    results_with_gt = [r for r in results if r.get("ground_truth_articles")]
    
    rt_precisions, rt_recalls = [], []
    ct_precisions, ct_recalls = [], []

    for r in results_with_gt:
        # Ground Truth pairs: (law_id, article_id)
        gt = set((art["law_id"], str(art["article_id"])) for art in r["ground_truth_articles"])
        
        # Context Docs pairs: (doc_id, article_id)
        ctx_list = r.get("context_docs", [])
        ctx_pairs = set((d["doc_id"], str(d["article_id"])) for d in ctx_list if d.get("doc_id") and d.get("article_id"))
        
        # Retrieval Performance
        tp_rt = len(ctx_pairs.intersection(gt))
        rt_precisions.append(tp_rt / len(ctx_pairs) if ctx_pairs else 0)
        rt_recalls.append(tp_rt / len(gt) if gt else 0)

        # Used Docs (Citation Performance)
        # Now used_docs is a list of dicts with doc_id and article_id
        used_list = r.get("used_docs", [])
        used_pairs = set((d["doc_id"], str(d["article_id"])) for d in used_list if d.get("doc_id") and d.get("article_id"))
        
        tp_ct = len(used_pairs.intersection(gt))
        ct_precisions.append(tp_ct / len(used_pairs) if used_pairs else 0)
        ct_recalls.append(tp_ct / len(gt) if gt else 0)

    def f1(p, r):
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    avg_rt_p = sum(rt_precisions) / len(rt_precisions) if rt_precisions else 0
    avg_rt_r = sum(rt_recalls) / len(rt_recalls) if rt_recalls else 0
    avg_ct_p = sum(ct_precisions) / len(ct_precisions) if ct_precisions else 0
    avg_ct_r = sum(ct_recalls) / len(ct_recalls) if ct_recalls else 0

    # Confusion Matrix for Đúng/Sai
    ds_results = [r for r in results if r['type'] == 'Đúng/Sai']
    tp = sum(1 for r in ds_results if r['reference_answer'] == 'Đúng' and r['predicted_answer'] == 'Đúng')
    tn = sum(1 for r in ds_results if r['reference_answer'] == 'Sai' and r['predicted_answer'] == 'Sai')
    fp = sum(1 for r in ds_results if r['reference_answer'] == 'Sai' and r['predicted_answer'] == 'Đúng')
    fn = sum(1 for r in ds_results if r['reference_answer'] == 'Đúng' and r['predicted_answer'] == 'Sai')

    # MCQ Error Distribution
    mc_results = [r for r in results if r['type'] == 'Trắc nghiệm' and not r['is_correct']]
    mc_errors = Counter(r['predicted_answer'] for r in mc_results)

    # Print Report
    print("# ALQAC-2025 Evaluation Metrics Report")
    
    print("\n## Answer Accuracy")
    print(f"- **Total Questions**: {total}")
    print(f"- **Overall Accuracy**: {correct/total:.4%} ({correct}/{total})")

    print("\n### Performance by Question Type")
    for t, stats in accuracy_by_type.items():
        print(f"- **{t}**: {stats['accuracy']:.4%} ({stats['correct']}/{stats['total']})")

    print("\n## Retrieval & Citation Performance")
    print("| Metric | Precision | Recall | F1 |")
    print("|---|---|---|---|")
    print(f"| **Retrieval (context_docs)** | {avg_rt_p:.4f} | {avg_rt_r:.4f} | {f1(avg_rt_p, avg_rt_r):.4f} |")
    print(f"| **Citation (used_docs)** | {avg_ct_p:.4f} | {avg_ct_r:.4f} | {f1(avg_ct_p, avg_ct_r):.4f} |")

    if ds_results:
        print("\n## Confusion Matrix (Đúng/Sai)")
        print("| | Predicted Đúng | Predicted Sai |")
        print("|---|---|---|")
        print(f"| **Actual Đúng** | {tp} (TP) | {fn} (FN) |")
        print(f"| **Actual Sai** | {fp} (FP) | {tn} (TN) |")

    if mc_results:
        print("\n## MCQ Error Distribution (Incorrect Predictions)")
        for opt, count in mc_errors.most_common():
            print(f"- **Option {opt}**: {count} errors")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute metrics for ALQAC-2025 evaluation results")
    parser.add_argument("--results", default="/home/nt-loi/law-chatbot/data/alqac25_eval_results_v3.json", help="Path to evaluation results JSON")
    args = parser.parse_args()
    
    compute_metrics(args.results)
