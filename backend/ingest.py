import json
import os
import re
from typing import List, Optional, Dict, Any
from sqlalchemy import text
from sqlmodel import Session, select
from tqdm import tqdm
from underthesea import word_tokenize
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVector, SparseVectorParams, SparseIndexParams
    )
from collections import Counter
from database import (
    PhapDienNode, PhapDienReference, PhapDienRelation,
    VBQPPLDoc, VBQPPLNode,
    engine, init_db
    )
from config import (
    PHAPDIEN_DIR, VBQPPL_DIR, 
    QDRANT_HOST, QDRANT_PORT, 
    EMBEDDING_MODEL, EMBEDDING_DIM
    )
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter

PHAP_DIEN_DIEU_PATH = os.path.join(PHAPDIEN_DIR, "Dieu.json")
PHAP_DIEN_LIENQUAN_PATH = os.path.join(PHAPDIEN_DIR, "LienQuan.json")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

def segment_text(text_str: str) -> str:
    return word_tokenize(text_str, format="text")


def compute_sparse_vector(text_str: str) -> SparseVector:
    """Compute sparse vector with unique indices (handles hash collisions)."""
    tokens = text_str.lower().split()
    token_counts = Counter(tokens)
    idx_to_value = {}
    for token, count in token_counts.items():
        idx = abs(hash(token)) % 100000
        idx_to_value[idx] = idx_to_value.get(idx, 0.0) + float(count)
    indices = list(idx_to_value.keys())
    values = list(idx_to_value.values())
    return SparseVector(indices=indices, values=values)


def extract_item_id_from_url(url: str) -> Optional[str]:
    match = re.search(r"ItemID=(\d+)", url)
    return match.group(1) if match else None

def extract_title_from_ref(ref_name: str) -> str:
    id_match = re.search(r"\d+/\d+/[A-ZĐ0-9\-]+", ref_name)
    if id_match:
        return id_match.group(0)
    match = re.search(r"^(?:\d+[\s\.]+\s*)?(.*)", ref_name)
    result = match.group(1).strip() if match else ref_name.strip()
    return result.strip("()")


def extract_label_from_ref(ref_name: str) -> str:
    text = ref_name.lstrip(" (0123456789. ")
    pattern = r"^(Điều|Chương|Mục|Khoản|Điểm|Phần)\s+([0-9a-z]+|[ivxlcdm]+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return f"{match.group(1)} {match.group(2)}".strip()
    return ""


def setup_qdrant_collections(client: QdrantClient, re_ingest=False):
    if re_ingest:
        client.delete_collection("phapdien")
        client.delete_collection("vbqppl")
    
    for name in ["phapdien", "vbqppl"]:
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config={"dense": VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))}
            )
            print(f"Created collection: {name}")


def ingest_vbqppl_docs():
    """Ingests all VBQPPL documents from vbqppl_dir using batch inserts."""
    print("\nIngesting VBQPPL documents...")
    crawled_files = [f for f in os.listdir(VBQPPL_DIR) if f.endswith(".json")]
    
    with engine.connect() as conn:
        # Skip already processed docs
        result = conn.execute(text("SELECT id FROM vbqppl_docs WHERE is_crawled = TRUE"))
        processed_ids = {row[0] for row in result}
        
        for filename in tqdm(crawled_files):
            doc_id = filename.replace(".json", "")
            if doc_id in processed_ids:
                continue
                
            filepath = os.path.join(VBQPPL_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            conn.execute(text("""
                INSERT INTO vbqppl_docs (id, url, is_crawled)
                VALUES (:id, :url, TRUE)
            """), {
                "id": doc_id,
                "url": data.get("url", "").strip()
            })
            conn.commit()

            # Batch insert nodes for this document
            sections = data.get("sections", [])
            if sections:
                node_data = [{
                    "doc_id": doc_id,
                    "label": s.get("label", "").strip(),
                    "content": s.get("content", "").strip(),
                    "is_structure": s.get("type") != "fallback"
                } for s in sections]
                
                conn.execute(text("""
                    INSERT INTO vbqppl_nodes (doc_id, label, content, is_structure)
                    VALUES (:doc_id, :label, :content, :is_structure)
                """), node_data)
            
            conn.commit()


def ingest_phapdien_nodes():
    """Ingests Phap Dien articles."""
    print("\nIngesting Phap Dien nodes...")
    with open(PHAP_DIEN_DIEU_PATH, "r", encoding="utf-8") as f:
        dieu_data = json.load(f)
    
    with engine.connect() as conn:
        # Get existing VBQPPL doc IDs
        result = conn.execute(text("SELECT id FROM vbqppl_docs"))
        existing_vbqppl_ids = {row[0] for row in result}
        
        # Batch Insert PhapDienNodes
        seen_mapcs = set() 
        batch_size = 1000
        for i in tqdm(range(0, len(dieu_data), batch_size)):
            batch = dieu_data[i:i+batch_size]
            items = []
            
            for item in batch:
                mapc = item.get("MAPC")
                if not mapc or mapc in seen_mapcs:
                    continue
                
                seen_mapcs.add(mapc) # Update immediately
                items.append({
                    "id": mapc,
                    "content": item.get("NoiDung", "").strip(),
                    "title": item.get("TEN", "").strip(),
                    "demuc_id": item.get("DeMucID"),
                })
            
            if items:
                conn.execute(text("""
                    INSERT INTO phapdien_nodes (id, content, title, demuc_id)
                    VALUES (:id, :content, :title, :demuc_id)
                """), items)
                conn.commit()
        
        # Ingest PhapDienReferences with placeholder handling
        print("\nIngesting PhapDienReferences...")
        added_placeholder_ids = set()
        for i in tqdm(range(0, len(dieu_data), batch_size)):
            batch = dieu_data[i:i+batch_size]
            refs_to_insert = []
            
            for item in batch:
                id = item.get("MAPC")
                if not id: 
                    continue
                
                for ref in item.get("VBQPPL", []):
                    link = ref.get("link", "").strip()
                    doc_id = extract_item_id_from_url(link)
                    if not doc_id: 
                        continue
                    
                    if doc_id in existing_vbqppl_ids:
                        conn.execute(text("""
                            UPDATE vbqppl_docs
                            SET title = :title
                            WHERE id = :doc_id
                        """), {
                            "doc_id": doc_id, 
                            "title": extract_title_from_ref(ref.get("name", ""))
                            })
                        conn.commit()

                    # Placeholder check
                    elif doc_id not in added_placeholder_ids:
                        conn.execute(text("""
                            INSERT INTO vbqppl_docs (id, title, url, is_crawled)
                            VALUES (:id, :title, :url, FALSE)
                        """), {
                            "id": doc_id,
                            "title": extract_title_from_ref(ref.get("name", "")),
                            "url": link
                        })
                        conn.commit()
                        added_placeholder_ids.add(doc_id)
                    
                    refs_to_insert.append({
                        "phapdien_id": id,
                        "vbqppl_doc_id": doc_id,
                        "vbqppl_label": extract_label_from_ref(ref.get("name", "")),
                        "details": ref.get("name", "")
                    })
            
            if refs_to_insert:
                conn.execute(text("""
                    INSERT INTO phapdien_references (phapdien_id, vbqppl_doc_id, vbqppl_label, details)
                    VALUES (:phapdien_id, :vbqppl_doc_id, :vbqppl_label, :details)
                """), refs_to_insert)
                conn.commit()


def ingest_phapdien_relations():
    """Ingests Phap Dien cross-references efficiently."""
    print("\nIngesting Phap Dien relations...")
    
    with open(PHAP_DIEN_LIENQUAN_PATH, "r", encoding="utf-8") as f:
        lienquan_data = json.load(f)
    
    with engine.connect() as conn:
        result = conn.execute(text("SELECT id FROM phapdien_nodes"))
        existing_ids = {row[0] for row in result}
    
    batch_size = 2000
    for i in tqdm(range(0, len(lienquan_data), batch_size)):
        batch = lienquan_data[i:i+batch_size]
        relations = [{
            "source_id": r.get("source_MAPC"),
            "target_id": r.get("target_MAPC")
        } for r in batch if r.get("source_MAPC") in existing_ids and r.get("target_MAPC") in existing_ids]
        
        if relations:
            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO phapdien_relations (source_id, target_id)
                    VALUES (:source_id, :target_id)
                    ON CONFLICT DO NOTHING
                """), relations)
                conn.commit()


def build_vector_index():
    """Builds Qdrant index for Phap Dien and VBQPPL nodes."""
    print("\nBuilding Vector Index...")
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"Using device: {device}")

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    setup_qdrant_collections(client, re_ingest=True)
    
    # Index Phap Dien Nodes
    print("Indexing Phap Dien nodes...")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT id, content, title, demuc_id FROM phapdien_nodes WHERE content IS NOT NULL AND content != ''"))
        rows = result.fetchall()
    
    batch_size = 8    
    for i in tqdm(range(0, len(rows), batch_size)):
        batch_rows = rows[i : i + batch_size]
        batch_texts = [segment_text(r[1]) for r in batch_rows]
        batch_embeddings = model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False)
        points = []
        for idx, row in enumerate(batch_rows):
            node_id, content, title, demuc_id = row
            segmented = batch_texts[idx]
            dense_vector = batch_embeddings[idx].tolist()
            sparse_vector = compute_sparse_vector(segmented)
            
            points.append(PointStruct(
                id=abs(hash(node_id)) % (2**63),
                vector={"dense": dense_vector, "sparse": sparse_vector},
                payload={"id": node_id, "title": title or "", "demuc_id": demuc_id or "", "source": "phapdien"}
            ))
        
        client.upsert(collection_name="phapdien", points=points)

    # for row in tqdm(rows):
    #     node_id, text_content, title, demuc_id = row
    #     segmented = segment_text(text_content)
    #     dense_vector = model.encode(segmented).tolist()
    #     sparse_vector = compute_sparse_vector(segmented)
        
    #     points.append(PointStruct(
    #         id=abs(hash(node_id)) % (2**63),
    #         vector={"dense": dense_vector, "sparse": sparse_vector},
    #         payload={"id": node_id, "title": title or "", "demuc_id": demuc_id or "", "source": "phapdien"}
    #     ))
        
    #     if len(points) >= batch_size:
    #         client.upsert(collection_name="phapdien", points=points)
    #         points = []
    
    # if points:
    #     client.upsert(collection_name="phapdien", points=points)
        
    # Index VBQPPL Nodes
    print("\nIndexing VBQPPL nodes...")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT id, doc_id, label, content, is_structure FROM vbqppl_nodes WHERE content IS NOT NULL AND content != ''"))
        rows = result.fetchall()
    
    batch_size = 8
    points = []
    
    for row in tqdm(rows):
        node_id, doc_id, label, content, is_structure = row
        
        # Determine if we need to chunk
        texts_to_index = [content] if is_structure else text_splitter.split_text(content)
        
        for i, raw_text in enumerate(texts_to_index):
            segmented = segment_text(raw_text)
            dense_vector = model.encode(segmented).tolist()
            sparse_vector = compute_sparse_vector(segmented)
            
            # Generate a unique ID for Qdrant (important if node is chunked)
            # Use original node_id if only 1 chunk, otherwise combine id and index
            qdrant_id = node_id if len(texts_to_index) == 1 else (node_id * 1000 + i)
            
            points.append(PointStruct(
                id=qdrant_id,
                vector={"dense": dense_vector, "sparse": sparse_vector},
                payload={
                    "id": node_id, 
                    "doc_id": doc_id, 
                    "label": label or "", 
                    "is_structure": is_structure, 
                    "chunk_index": i, # Track which chunk this is
                    "source": "vbqppl"
                }
            ))

            if len(points) >= batch_size:
                client.upsert(collection_name="vbqppl", points=points)
                points = []
    
    if points:
        client.upsert(collection_name="vbqppl", points=points)
    
def main():
    print("\nInitializing Database...")
    init_db(drop_all=True)
    
    ingest_vbqppl_docs()
    ingest_phapdien_nodes()
    ingest_phapdien_relations()
    build_vector_index()
    
    print("\nData Ingestion Complete!")


if __name__ == "__main__":
    main()
