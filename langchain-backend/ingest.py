from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, PointStruct
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from fastembed import SparseTextEmbedding
import uuid
from tqdm import tqdm
import json
import torch
import re

load_dotenv()

def slugify_model_name(model_name: str) -> str:
    """Converts model name to a slug suitable for collection names."""
    slug = model_name.split("/")[-1] # Take the name after the org
    slug = re.sub(r'[^a-zA-Z0-9]', '_', slug).lower()
    return slug.strip("_")

def get_collection_name(source: str, model_name: str) -> str:
    """Generates a collection name based on source and model."""
    return f"{source}_{slugify_model_name(model_name)}"

model_name = os.getenv("EMBEDDING_MODEL")
model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
sparse_embedding = SparseTextEmbedding(model_name="Qdrant/bm25")

client = QdrantClient(path="./qdrant_data")
pd_collection = get_collection_name("phapdien", model_name)
# vb_collection = f"vbqppl_nodes_{embedding_name}"
vector_size = os.getenv("VECTOR_SIZE")

if client.collection_exists(pd_collection):
    client.delete_collection(pd_collection)

# if client.collection_exists(vb_collection):
#     client.delete_collection(vb_collection)

client.create_collection(
    collection_name=pd_collection,
    vectors_config={"dense": VectorParams(size=vector_size, distance=Distance.COSINE)},
    sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))}
)
print(f"Created collection: {pd_collection} (Dim: {vector_size})")

with open(os.getenv("PHAPDIEN_DIR") + "/Dieu.json", "r", encoding="utf-8") as f:
    data = json.load(f)

points = []

batch_size = 256
for i in tqdm(range(0, len(data), batch_size), desc="Indexing Phap Dien"):
    batch_data = data[i:i + batch_size]
    batch_texts = [f"{item['TEN']} {item['NoiDung']}" for item in batch_data]
    batch_dense_vectors = embedding.embed_documents(batch_texts)
    batch_sparse_vectors = sparse_embedding.embed(batch_texts)
    
    for item, dense_vec, sparse_vec in zip(batch_data, batch_dense_vectors, batch_sparse_vectors):
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
                    "id": item["ID"],
                    "source": "phapdien",
                    "content": f"{item['TEN']} {item['NoiDung']}"
                }
            )
        )

client.upsert(
    collection_name=pd_collection,
    points=points
)

print("Indexed Phap Dien nodes into", pd_collection)
client.close()