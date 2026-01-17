from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, PointStruct
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
import uuid
from tqdm import tqdm
import json
import torch

from utils import get_alqac_point_id

load_dotenv()

# Configuration
batch_size = 128
model_name = os.getenv("EMBEDDING_MODEL")
model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
encode_kwargs = {"convert_to_numpy": True, "normalize_embeddings": True}

print(f"Loading dense embedding model: {model_name}")
embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
embedding._client.max_seq_length = int(os.getenv("MAX_SEQ_LENGTH"))

print("Loading sparse embedding model: Qdrant/bm25")
sparse_embedding = SparseTextEmbedding(model_name="Qdrant/bm25")

client = QdrantClient(host=os.getenv("QDRANT_HOST"), port=os.getenv("QDRANT_PORT"))
collection_name = "alqac25_collection"
vector_size = int(os.getenv("VECTOR_SIZE"))

# Recreate collection
if client.collection_exists(collection_name):
    print(f"Deleting existing collection: {collection_name}")
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config={"dense": VectorParams(size=vector_size, distance=Distance.COSINE)},
    sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))}
)
print(f"Created collection: {collection_name} (Dim: {vector_size})")

# Load data
data_path = "/home/nt-loi/law-chatbot/ALQAC-2025/alqac25_law.json"
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

points = []
batch_texts = []
batch_metas = []

for doc in tqdm(data, desc="Processing documents"):
    doc_id = doc.get("id")
    articles = doc.get("articles", [])
    
    for article in articles:
        article_id = article.get("id")
        content = article.get("text", "")
        # Use doc_id as title since it's the law name in ALQAC-2025
        title = doc_id
        
        embed_content = f"{title}\n{content}"
        
        batch_texts.append(embed_content)
        batch_metas.append({
            "doc_id": doc_id,
            "article_id": article_id,
            "title": title,
            "content": content,
            "embed_content": embed_content
        })
        
        if len(batch_texts) >= batch_size:
            # Embed batch
            batch_dense_vectors = embedding.embed_documents(batch_texts)
            batch_sparse_vectors = list(sparse_embedding.embed(batch_texts))
            
            for text, meta, dense_vec, sparse_vec in zip(batch_texts, batch_metas, batch_dense_vectors, batch_sparse_vectors):
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "dense": dense_vec,
                            "sparse": {
                                "indices": sparse_vec.indices,
                                "values": sparse_vec.values
                            }
                        },
                        payload={
                            "id": get_alqac_point_id(meta["doc_id"], meta["article_id"]),
                            "doc_id": meta["doc_id"],
                            "article_id": meta["article_id"],
                            "source": "alqac25",
                            "title": meta["title"],
                            "content": meta["content"],
                            "embed_content": text
                        }
                    )
                )
            
            client.upsert(collection_name=collection_name, points=points)
            points = []
            batch_texts = []
            batch_metas = []

# Process remaining items
if batch_texts:
    batch_dense_vectors = embedding.embed_documents(batch_texts)
    batch_sparse_vectors = list(sparse_embedding.embed(batch_texts))
    
    for text, meta, dense_vec, sparse_vec in zip(batch_texts, batch_metas, batch_dense_vectors, batch_sparse_vectors):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vec,
                    "sparse": {
                        "indices": sparse_vec.indices,
                        "values": sparse_vec.values
                    }
                },
                payload={
                    "id": get_alqac_point_id(meta["doc_id"], meta["article_id"]),
                    "doc_id": meta["doc_id"],
                    "article_id": meta["article_id"],
                    "source": "alqac25",
                    "title": meta["title"],
                    "content": meta["content"],
                    "embed_content": text
                }
            )
        )
    client.upsert(collection_name=collection_name, points=points)

print(f"Indexed ALQAC-2025 nodes into {collection_name}")
client.close()
