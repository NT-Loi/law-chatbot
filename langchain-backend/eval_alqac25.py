import csv
import json
import asyncio
import os
import logging
import argparse
import ast
import re
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

import hashlib
from rag import RAG
from chat import LegalRAGChain, clean_reasoning_output
from prompts import ALQAC_ANSWER_SYSTEM_PROMPT
from utils import get_alqac_point_id

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation_alqac25.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

async def evaluate_alqac25(input_file, output_file, limit=None):
    if not os.path.exists(input_file):
        logging.error(f"Input file not found: {input_file}")
        return

    logging.info(f"Loading dataset from {input_file}...")
    if input_file.endswith('.json'):
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    else:
        dataset = []
        with open(input_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset.append(row)
    
    # Load ground truth map (for retrieval eval) from the same dataset if it has relevant_articles
    ground_truth_map = {}
    for item in dataset:
        qid = item.get("question_id")
        if "relevant_articles" in item:
            ground_truth_map[qid] = item.get("relevant_articles", [])
    
    if limit:
        dataset = dataset[:limit]
        logging.info(f"Limiting evaluation to first {limit} items.")

    logging.info("Initializing RAG engine...")
    try:
        rag_engine = RAG()
    except Exception as e:
        logging.error(f"Failed to initialize RAG engine: {e}")
        return

    logging.info("Initializing LegalRAGChain...")
    try:
        chain = LegalRAGChain()
    except Exception as e:
        logging.error(f"Failed to initialize LegalRAGChain: {e}")
        rag_engine.close()
        return

    results = []
    processed_ids = set()
    
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                for r in existing_data:
                    processed_ids.add(r.get("question_id"))
            results = existing_data
            logging.info(f"Resuming from existing output. Found {len(processed_ids)} processed items.")
        except Exception as e:
            logging.warning(f"Failed to load existing output for resume: {e}. Starting fresh.")

    logging.info(f"Starting evaluation on {len(dataset)} items...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    collection_names = ["alqac25_collection"]

    for i, item in enumerate(tqdm(dataset, desc="Evaluating ALQAC25")):
        qid = item.get("question_id")
        if qid in processed_ids:
            continue
            
        q_type = item.get("question_type") or item.get("type")
        q_content = item.get("text") or item.get("question_content") or item.get("question")
        reference_answer = item.get("answer") or item.get("answer_content") or item.get("reference_answer")
        
        # Prepare the query and user message
        if q_type == "Đúng/Sai":
            user_msg = q_content
            query = q_content
        elif q_type == "Trắc nghiệm":
            choices = item.get("choices", {})
            choices_str = "\n".join([f"{k}: {v}" for k, v in choices.items()])
            user_msg = f"{q_content}\n\nCác phương án:\n{choices_str}"
        else:
            user_msg = q_content

        system_response = ""
        context_docs = []
        used_docs = []
        
        try:
            # Call the chat chain with specified collection and specialized system prompt
            async for chunk_str in chain.chat(user_msg, [], rag_engine, collection_names=collection_names, system_prompt=ALQAC_ANSWER_SYSTEM_PROMPT):
                try:
                    chunk = json.loads(chunk_str)
                    type_ = chunk.get("type")
                    
                    if type_ == "content":
                        delta = chunk.get("delta", "")
                        system_response += delta
                    elif type_ == "sources":
                        data = chunk.get("data", [])
                        for doc in data:
                            context_docs.append(doc)
                    elif type_ == "used_docs":
                        # data might be rich docs or ids
                        received_data = chunk.get("data", []) or []
                        received_ids = chunk.get("ids", []) or []
                        
                        target_ids = received_ids
                        if received_data:
                            target_ids = [d.get("id") for d in received_data if d.get("id")]
                        
                        for uid in target_ids:
                            # Clean ID (strip prefixes model might have added)
                            clean_uid = re.sub(r'^(doc_|DOC_|doc-|DOC-)', '', uid).strip()
                            
                            found_in_ctx = False
                            for cd in context_docs:
                                ctx_id = cd.get("id", "")
                                if ctx_id == clean_uid or ctx_id == uid:
                                    used_docs.append(cd)
                                    found_in_ctx = True
                                    break
                            
                            if not found_in_ctx:
                                # Fallback to received data or just the ID
                                fallback_doc = {"id": uid}
                                for rd in received_data:
                                    if rd.get("id") == uid:
                                        fallback_doc = rd
                                        break
                                used_docs.append(fallback_doc)
                    elif type_ == "error":
                         logging.warning(f"Error from chain for {qid}: {chunk.get('content')}")
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            logging.error(f"Exception processing {qid}: {e}")
            system_response = f"[EXCEPTION: {str(e)}]"
        
        # Clean response
        system_response = clean_reasoning_output(system_response)

        # Extraction from <USED_DOCS> is now handled by the chain stream (stream_with_citations)

        # Extract the final answer
        predicted_answer = system_response.strip().replace(".", "").replace(":", "")
        if q_type == "Đúng/Sai":
            if "Đúng" in predicted_answer[:10]:
                predicted_answer = "Đúng"
            elif "Sai" in predicted_answer[:10]:
                predicted_answer = "Sai"
        elif q_type == "Trắc nghiệm":
            match = re.search(r'\b([A-D])\b', predicted_answer)
            if match:
                predicted_answer = match.group(1)
            else:
                predicted_answer = predicted_answer[:1]
        else:
            # Tự luận - keep as is for semantic comparison later
            predicted_answer = system_response.strip()

        # Check correctness
        if q_type == "Tự luận":
            # Simple normalized comparison for essay
            correct = predicted_answer.lower().replace(" ", "") == reference_answer.lower().replace(" ", "")
        else:
            correct = predicted_answer == reference_answer

        results.append({
            "question_id": qid,
            "type": q_type,
            "question": q_content,
            "reference_answer": reference_answer,
            "predicted_answer": predicted_answer,
            "is_correct": correct,
            "system_full_response": system_response,
            "context_docs": [
                {"id": d.get("id"), "doc_id": d.get("doc_id"), "article_id": d.get("article_id")} 
                for d in context_docs
            ],
            "used_docs": [
                {"id": d.get("id"), "doc_id": d.get("doc_id"), "article_id": d.get("article_id")}
                for d in used_docs
            ],
            "ground_truth_articles": ground_truth_map.get(qid, [])
        })
        
        # Save periodically
        if (i + 1) % 10 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

    # Final save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    # Calculate Accuracy
    if results:
        correct_count = sum(1 for r in results if r["is_correct"])
        accuracy = correct_count / len(results)
        logging.info(f"Evaluation complete. Accuracy: {accuracy:.4f} ({correct_count}/{len(results)})")
    
    logging.info(f"Results saved to {output_file}")
    
    try:
        rag_engine.close()
    except:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ALQAC 2025 Dataset")
    parser.add_argument("--input", default="../ALQAC-2025/ALQAC_2025_QA.csv", help="Path to input dataset CSV")
    parser.add_argument("--output", default="../data/alqac25_eval_results.json", help="Path to output results JSON")
    parser.add_argument("--limit", type=int, help="Limit number of questions to evaluate", default=None)
    
    args = parser.parse_args()
    
    asyncio.run(evaluate_alqac25(args.input, args.output, args.limit))

