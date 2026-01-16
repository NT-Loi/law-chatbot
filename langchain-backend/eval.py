import json
import asyncio
import os
import logging
import argparse
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

# Ensure we are running from the correct directory or path is set up
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag import RAG
from chat import LegalRAGChain

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

async def evaluate(input_file, output_file, limit=None):
    if not os.path.exists(input_file):
        logging.error(f"Input file not found: {input_file}")
        return

    logging.info(f"Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
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
    
    logging.info(f"Starting evaluation on {len(dataset)} items...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for i, item in enumerate(tqdm(dataset, desc="Evaluating")):
        question = item.get("question")
        reference_answer = item.get("answer")
        reference_refs = item.get("reference", [])
        
        system_response = ""
        used_docs = []
        context_docs = []
        
        try:
            # Call the chat chain
            # history is empty list
            # Note: LegalRAGChain.chat yields strings representing JSON objects
            async for chunk_str in chain.chat(question, [], rag_engine):
                try:
                    chunk = json.loads(chunk_str)
                    type_ = chunk.get("type")
                    
                    if type_ == "content":
                        delta = chunk.get("delta", "")
                        system_response += delta
                    elif type_ == "sources":
                        data = chunk.get("data", [])
                        # This is the context after reranking
                        for doc in data:
                            context_docs.append({
                                "id": doc.get("id"),
                                "doc_id": doc.get("doc_id"),
                                "title": doc.get("title"),
                                "hierarchy_path": doc.get("hierarchy_path") or doc.get("title", ""),
                                "content": doc.get("content"),
                                "score": doc.get("score"),
                                "rerank_score": doc.get("rerank_score")
                            })
                    elif type_ == "used_docs":
                        # This is the documents actually cited by the model
                        data = chunk.get("data", []) or chunk.get("ids", [])
                        # Data might be list of dicts (rich) or list of strings (ids)
                        if data and isinstance(data[0], dict):
                             for doc in data:
                                used_docs.append({
                                    "id": doc.get("id"),
                                    "doc_id": doc.get("doc_id"),
                                    "title": doc.get("title"),
                                    "hierarchy_path": doc.get("hierarchy_path") or doc.get("title", ""),
                                    "content": doc.get("content"),
                                    "score": doc.get("score"),
                                    "rerank_score": doc.get("rerank_score")
                                })
                        else:
                             # Just IDs
                             used_docs = [{"id": uid} for uid in data]

                    elif type_ == "error":
                         logging.warning(f"Error from chain for q='{question}': {chunk.get('content')}")
                         system_response += f"\n[ERROR: {chunk.get('content')}]"
                except json.JSONDecodeError:
                    logging.warning(f"Failed to decode chunk: {chunk_str}")

        except Exception as e:
            logging.error(f"Exception processing question {i}: {question}. Error: {e}")
            system_response = f"[EXCEPTION: {str(e)}]"
        
        results.append({
            "id": i,
            "question": question,
            "reference_answer": reference_answer,
            "reference_refs": reference_refs,
            "system_response": system_response,
            "context_docs": context_docs,
            "used_docs": used_docs
        })
        
        # Save periodically to avoid losing all data if crash
        if (i + 1) % 10 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

    # Final save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    logging.info(f"Evaluation complete. Results saved to {output_file}")
    
    # Close RAG connection
    try:
        rag_engine.close()
    except:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Legal Chatbot System")
    parser.add_argument("--input", default="../data/du_lieu_luat_dataset.json", help="Path to input dataset JSON")
    parser.add_argument("--output", default="../data/evaluation_results.json", help="Path to output results JSON")
    parser.add_argument("--limit", type=int, help="Limit number of questions to evaluate (for testing)", default=None)
    
    args = parser.parse_args()
    
    asyncio.run(evaluate(args.input, args.output, args.limit))
