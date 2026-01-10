from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, SparseVector, Fusion, FusionQuery, Filter, FieldCondition, MatchValue

from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding

import torch
from typing import Any
from utils import get_collection_name

import os
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

class RAG:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Device: {self.device}")

        # self.model = init_chat_model("google_genai:gemini-2.0-flash")
        # logging.info(f"Initialize LLM: {self.model}")
        
        self.qdrant_client = QdrantClient(path=os.getenv("QDRANT_PATH"))
        logging.info(f"Initialize Qdrant client: {self.qdrant_client}")
        
        model_name = os.getenv("EMBEDDING_MODEL")
        model_kwargs = {"device": self.device}
        encode_kwargs = {"normalize_embeddings": False}
        self.embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        logging.info(f"Initialize embedding: {self.embedding}")
        
        self.sparse_embedding = SparseTextEmbedding(model_name="Qdrant/bm25")
        logging.info(f"Initialize sparse embedding: {self.sparse_embedding}")
        
        self.pd_collection_name = get_collection_name("phapdien", model_name)
        
    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        logging.info(f"Retrieve documents for query: {query}")
        dense_vec = self.embedding.embed_query(query)
        sparse_emb = next(self.sparse_embedding.embed([query]))
        sparse_vec = SparseVector(
            indices=sparse_emb.indices.tolist(),
            values=sparse_emb.values.tolist(),
        )
        results = self.qdrant_client.query_points(
            collection_name=self.pd_collection_name,
            prefetch=[
                Prefetch(query=dense_vec, using="dense", limit=top_k * 3),
                Prefetch(query=sparse_vec, using="sparse", limit=top_k * 3)
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k
        ).points

        sources = []
        for i, result in enumerate(results):
            payload = result.payload
            source_id = payload.get("id")
            content = payload.get("content")
                    
            sources.append({"id": source_id, "content": content})
        return sources

    def rerank(self, query: str, sources: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        logging.info(f"Rerank documents for query: {query}")
        pass
        # return sources

    # def build_context(self, sources: list[dict[str, Any]]) -> str:
    #     docs_content = "\n\n".join(doc["content"] for doc in sources)
    #     return docs_content

    def close(self):
        self.qdrant_client.close()