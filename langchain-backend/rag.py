import os
import torch
import logging
from typing import Any, List
from dotenv import load_dotenv
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, SparseVector, Fusion, FusionQuery, Filter, FieldCondition, MatchAny
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
import voyageai

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
        self.rerank_model_name = os.getenv("RERANKING_MODEL", "rerank-2")
        if "/" in self.rerank_model_name:
             logging.warning(f"RERANKING_MODEL ({self.rerank_model_name}) looks like a path/RepoID. Voyage requires a model name (e.g. 'rerank-2'). Defaulting to 'rerank-2'.")
             self.rerank_model_name = "rerank-2"
        
        self.voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
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
                limit=top_k
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
        # results_pd = self._query_collection(self.pd_collection_name, dense_vec, sparse_vec, top_k)
        results_vb = self._query_collection(self.vb_collection_name, dense_vec, sparse_vec, top_k)

        # 3. Standardize results
        sources = []
        seen_point_ids = set()

        # for point in results_pd + results_vb:
        for point in results_vb:
            payload = point.payload
            
            id = payload.get("id", "")
            # Deduplication
            if id in seen_point_ids:
                continue
            seen_point_ids.add(id)

            sources.append({
                "id": id,
                "doc_id": payload.get("doc_id", ""),
                "title": payload.get("title", ""),
                "hierarchy_path": payload.get("hierarchy_path", ""),
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

        # Prepare documents for Voyage (list of strings)
        documents = [doc.get("content", "") for doc in sources]
        
        try:
            results = self.voyage_client.rerank(
                query=query,
                documents=documents,
                model=self.rerank_model_name,
                top_k=top_k
            )
            
            # results.results contains the ranked items
            # Each item has .index and .relevance_score
            scored_sources = []
            for r in results.results:
                doc = sources[r.index].copy()
                doc["rerank_score"] = r.relevance_score
                scored_sources.append(doc)
            
            # logging.info(f"Voyage Reranking: {scored_sources}")

            return scored_sources

        except Exception as e:
            logging.error(f"Voyage Reranking failed: {e}")
            # Fallback: return top_k of original sources (assuming they were roughly ordered by retrieval)
            return sources[:top_k]

    def get_documents_by_ids(self, ids: List[str]) -> List[dict]:
        """
        Fetch full document content from Qdrant based on a list of IDs.
        Separates IDs by collection based on the presence of a hyphen.
        """
        pd_ids = [i for i in ids if "-" in i]
        vb_ids = [i for i in ids if "-" not in i]
        
        results = []
        
        def fetch(collection, id_list):
            if not id_list: return []
            try:
                # Use scroll to retrieve points filtering by 'id' payload field
                points, _ = self.qdrant_client.scroll(
                    collection_name=collection,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="id",
                                match=MatchAny(any=id_list)
                            )
                        ]
                    ),
                    limit=len(id_list),
                    with_payload=True,
                    with_vectors=False
                )
                return points
            except Exception as e:
                logging.error(f"Error fetching docs from {collection}: {e}")
                return []

        pd_points = fetch(self.pd_collection_name, pd_ids)
        vb_points = fetch(self.vb_collection_name, vb_ids)
        
        for point in pd_points + vb_points:
             payload = point.payload
             results.append({
                "id": payload.get("id", ""),
                "doc_id": payload.get("doc_id", ""),
                "title": payload.get("title", ""),
                "hierarchy_path": payload.get("hierarchy_path", ""),
                "url": payload.get("url", "#"),
                "content": payload.get("content", ""),
                "source": payload.get("source", "")
            })
            
        return results

    def close(self):
        self.qdrant_client.close()