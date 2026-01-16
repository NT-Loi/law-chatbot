"""
Ingest Pháp Điển and VBQPPL data into PostgreSQL
This provides fast document retrieval for UI display (replacing slow Qdrant payload reads)
"""
import json
import os
import hashlib
from tqdm import tqdm
from sqlmodel import Session, select
from sqlalchemy import text
from dotenv import load_dotenv

from models import (
    engine, init_db,
    VBQPPLDoc, VBQPPLSection,
    PhapDienDieu
)

load_dotenv()

BATCH_SIZE = 100  # Smaller batch for stability


def get_section_id(doc_id: str, hierarchy_path: str) -> str:
    """
    Generate unique section ID (MD5 Hash) based on document ID and section path.
    This matches the ID format used in Qdrant for consistency.
    """
    safe_doc_id = str(doc_id) if doc_id else "unknown_doc"
    safe_path = str(hierarchy_path) if hierarchy_path else "general"
    raw_combination = f"{safe_doc_id}_{safe_path}"
    return hashlib.md5(raw_combination.encode('utf-8')).hexdigest()


def ingest_vbqppl(session: Session, data_path: str):
    """Ingest VBQPPL documents and sections"""
    print(f"\n📚 Loading VBQPPL data from: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   Found {len(data)} documents")
    
    doc_count = 0
    section_count = 0
    errors = 0
    
    for item in tqdm(data, desc="Processing VBQPPL"):
        doc_id = item.get("id")
        if not doc_id:
            continue
        
        try:
            # Create document
            doc = VBQPPLDoc(
                id=doc_id,
                title=item.get("title") if item.get("title") != "Unknown Title" else None,
                url=item.get("url"),
                content=item.get("content"),
                status=item.get("status"),
                error_message=item.get("error_message"),
                original_name=item.get("original_name"),
                original_link=item.get("original_link")
            )
            session.add(doc)
            session.flush()  # Flush to get any constraint errors
            doc_count += 1
            
            # Create sections
            sections = item.get("sections") or []
            for section in sections:
                hierarchy_path = section.get("hierarchy_path", "")
                
                # Calculate Hash ID to match Qdrant ID
                hash_id = get_section_id(doc_id, hierarchy_path)
                
                section_obj = VBQPPLSection(
                    hash_id=hash_id,
                    doc_id=doc_id,
                    label=section.get("label", ""),
                    content=section.get("content", ""),
                    hierarchy_path=hierarchy_path,
                    section_type=section.get("type")
                )
                session.add(section_obj)
                section_count += 1
            
            # Commit every BATCH_SIZE documents
            if doc_count % BATCH_SIZE == 0:
                session.commit()
                
        except Exception as e:
            session.rollback()
            errors += 1
            if errors <= 5:  # Only print first 5 errors
                print(f"\n⚠️  Error ingesting {doc_id}: {str(e)[:100]}")
            continue
    
    # Final commit
    try:
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"\n⚠️  Error in final commit: {str(e)[:200]}")
    
    print(f"✅ Ingested {doc_count} VBQPPL documents with {section_count} sections ({errors} errors)")


def ingest_phapdien(session: Session, data_path: str):
    """Ingest Pháp Điển data"""
    print(f"\n📖 Loading Pháp Điển data from: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   Found {len(data)} điều")
    
    count = 0
    errors = 0
    
    for item in tqdm(data, desc="Processing Pháp Điển"):
        try:
            dieu = PhapDienDieu(
                id=item.get("ID"),
                chi_muc=item.get("ChiMuc"),
                mapc=item.get("MAPC"),
                ten=item.get("TEN", ""),
                noi_dung=item.get("NoiDung", ""),
                chu_de_id=item.get("ChuDeID"),
                de_muc_id=item.get("DeMucID"),
                chuong_mapc=item.get("ChuongMAPC"),
                stt=item.get("STT"),
                vbqppl_refs=json.dumps(item.get("VBQPPL", []), ensure_ascii=False) if item.get("VBQPPL") else None
            )
            session.add(dieu)
            count += 1
            
            if count % BATCH_SIZE == 0:
                session.commit()
                
        except Exception as e:
            session.rollback()
            errors += 1
            if errors <= 5:
                print(f"\n⚠️  Error ingesting {item.get('ID')}: {str(e)[:100]}")
            continue
    
    # Final commit
    try:
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"\n⚠️  Error in final commit: {str(e)[:200]}")
    
    print(f"✅ Ingested {count} Pháp Điển điều ({errors} errors)")


def main():
    import sys
    
    # Check for --drop flag
    drop_all = "--drop" in sys.argv
    
    print("🚀 Starting PostgreSQL Ingestion")
    print("=" * 50)
    
    # Initialize database
    init_db(drop_all=drop_all)
    
    # Get data paths from environment
    vbqppl_path = os.getenv("VBQPPL", "data/vbqppl_content.json")
    phapdien_dir = os.getenv("PHAPDIEN_DIR", "data/phap_dien")
    phapdien_path = os.path.join(phapdien_dir, "Dieu.json")
    
    # Make paths absolute if needed
    if not os.path.isabs(vbqppl_path):
        vbqppl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), vbqppl_path)
    if not os.path.isabs(phapdien_path):
        phapdien_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), phapdien_path)
    
    with Session(engine) as session:
        # Ingest VBQPPL
        if os.path.exists(vbqppl_path):
            ingest_vbqppl(session, vbqppl_path)
        else:
            print(f"⚠️  VBQPPL file not found: {vbqppl_path}")
        
        # Ingest Pháp Điển
        if os.path.exists(phapdien_path):
            ingest_phapdien(session, phapdien_path)
        else:
            print(f"⚠️  Pháp Điển file not found: {phapdien_path}")
    
    print("\n" + "=" * 50)
    print("✅ PostgreSQL Ingestion Complete!")


if __name__ == "__main__":
    main()
