import os
import torch
import logging
from typing import Any, List
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, SparseVector, Fusion, FusionQuery
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import get_collection_name

import hashlib

def get_point_id(doc_id: str, hierarchy_path: str) -> str:
    """
    Tạo ID duy nhất (MD5 Hash) cho point dựa trên ID văn bản và vị trí điều khoản.
    Output ví dụ: 'a1b2c3d4e5f6...' (32 ký tự, an toàn cho URL/Frontend key)
    """
    # Xử lý trường hợp null
    safe_doc_id = str(doc_id) if doc_id else "unknown_doc"
    safe_path = str(hierarchy_path) if hierarchy_path else "general"

    # Tạo chuỗi kết hợp (Raw String)
    raw_combination = f"{safe_doc_id}_{safe_path}"

    # Hash MD5 để tạo chuỗi ID cố định, không trùng lặp và an toàn
    return hashlib.md5(raw_combination.encode('utf-8')).hexdigest()

load_dotenv()
logging.basicConfig(level=logging.INFO)

class RAG:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"RAG Device: {self.device}")
        
        # Init Qdrant
        self.qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST"), 
            port=int(os.getenv("QDRANT_PORT"))
        )
        
        # Init Dense Embedding
        embedding_name = os.getenv("EMBEDDING_MODEL")
        model_kwargs = {"device": self.device}
        encode_kwargs = {"convert_to_numpy": True, "normalize_embeddings": True}
        self.embedding = HuggingFaceEmbeddings(
            model_name=embedding_name, 
            model_kwargs=model_kwargs, 
            encode_kwargs=encode_kwargs
        )
        # Fix max length explicitly if needed, or rely on model default
        self.embedding._client.max_seq_length = int(os.getenv("MAX_SEQ_LENGTH", 512))
        
        # Init Sparse Embedding
        self.sparse_embedding = SparseTextEmbedding(model_name="Qdrant/bm25")
        
        # Collection Names
        self.pd_collection_name = get_collection_name("phapdien", embedding_name)
        self.vb_collection_name = get_collection_name("vbqppl", embedding_name)

        # Init Reranker
        rerank_model_name = os.getenv("RERANKING_MODEL")
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_name)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_name)
        self.rerank_model.to(self.device)
        self.rerank_model.eval()
        logging.info("RAG Components Initialized Successfully.")

    def _query_collection(self, collection_name: str, dense_vec: List[float], sparse_vec: SparseVector, top_k: int):
        """Helper to query a single collection with Hybrid Search (RRF)"""
        try:
            return self.qdrant_client.query_points(
                collection_name=collection_name,
                prefetch=[
                    Prefetch(query=dense_vec, using="dense", limit=top_k * 5),
                    Prefetch(query=sparse_vec, using="sparse", limit=top_k * 5)
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k * 2 # Get slightly more for reranking
            ).points
        except Exception as e:
            logging.warning(f"Failed to query collection {collection_name}: {e}")
            return []

    def retrieve(self, query: str, top_k: int = 5) -> List[dict[str, Any]]:
        # 1. Embed Query
        dense_vec = self.embedding.embed_query(query)
        sparse_gen = self.sparse_embedding.embed([query])
        sparse_emb = next(sparse_gen)
        sparse_vec = SparseVector(
            indices=sparse_emb.indices.tolist(),
            values=sparse_emb.values.tolist(),
        )

        # 2. Retrieve from both collections
        results_pd = self._query_collection(self.pd_collection_name, dense_vec, sparse_vec, top_k)
        results_vb = self._query_collection(self.vb_collection_name, dense_vec, sparse_vec, top_k)

        # 3. Standardize results
        sources = []
        # seen_ids = set() # <-- BỎ hoặc sửa logic Deduplication này
        # Vì bây giờ ta muốn lấy nhiều đoạn khác nhau trong cùng 1 văn bản,
        # nên không được filter theo doc_id nữa.
        
        # Dùng set để lọc các đoạn trùng nhau hoàn toàn (cùng doc + cùng hierarchy)
        seen_point_ids = set()

        for point in results_pd + results_vb:
            payload = point.payload
            doc_id = payload.get("id")
            hierarchy_path = payload.get("hierarchy_path", "")
            
            # Tạo Point ID mới
            point_id = get_point_id(doc_id, hierarchy_path)

            # Deduplication: Chỉ bỏ qua nếu ĐÚNG ĐOẠN ĐÓ đã xuất hiện rồi
            if point_id in seen_point_ids:
                continue
            seen_point_ids.add(point_id)

            sources.append({
                "id": point_id, # <--- Dùng ID mới sinh ra
                "original_doc_id": doc_id, # <--- Nên giữ lại ID gốc để FE hiển thị hoặc gọi API get detail
                "title": payload.get("title", ""),
                "hierarchy_path": hierarchy_path,
                "url": payload.get("url", "#"),
                "content": payload.get("content", ""),
                "score": point.score,
                "source": payload.get("source", "")
            })
        
        return sources
        # 4. Rerank
        # return self.rerank(query, sources, top_k=top_k)

    def rerank(self, query: str, sources: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        if not sources:
            return []

        # Prepare pairs: [Query, Content]
        pairs = [[query, doc["content"]] for doc in sources]
        
        MAX_LENGTH = 2304 
        
        with torch.no_grad():
            inputs = self.rerank_tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=MAX_LENGTH
            ).to(self.device)
            
            scores = self.rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
            
        scored_sources = []
        for i, score in enumerate(scores):
            doc = sources[i].copy()
            doc["rerank_score"] = score.item()
            scored_sources.append(doc)
            
        # Sort by rerank score desc
        scored_sources.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return scored_sources[:top_k]

    def close(self):
        self.qdrant_client.close()