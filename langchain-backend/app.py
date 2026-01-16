from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from qdrant_client.models import Filter, FieldCondition, MatchValue
from contextlib import asynccontextmanager
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from typing import List, Optional
from pydantic import BaseModel
import re
import json

from chat import ChatRouter, LegalRAGChain, WebLawChain, ChitChatChain, HybridChain, ChatMode
from rag import RAG
from models import get_async_session, VBQPPLDoc, VBQPPLSection, PhapDienDieu

logging.basicConfig(level=logging.INFO)

# Global variables
router = None
legal_rag_chain = None
web_chain = None
chit_chat_chain = None
rag_engine = None
hybrid_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global router, legal_rag_chain, web_chain, chit_chat_chain, rag_engine
    
    # 1. Init RAG Engine (Heavy loading)
    try:
        logging.info("Initializing RAG Engine...")
        rag_engine = RAG()
        logging.info("RAG Engine Initialized.")
    except Exception as e:
        logging.error(f"FATAL: Failed to initialize RAG: {str(e)}")
        raise

    # 2. Init Chat Strategies
    try:
        router = ChatRouter()
        legal_rag_chain = LegalRAGChain()
        web_chain = WebLawChain()
        hybrid_chain = HybridChain()
        chit_chat_chain = ChitChatChain()
        logging.info("Chat Strategies Initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize Chat Chains: {str(e)}")
        raise
    
    yield
    
    # Cleanup
    if rag_engine:
        rag_engine.close()
        logging.info("RAG Engine Closed.")

app = FastAPI(
    title="Vietnamese Law RAG Chatbot",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Law RAG API is ready"}

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []
    mode: ChatMode = ChatMode.AUTO 
    stream: bool = True # Flag to control streaming vs full response

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine not ready")

    try:
        # 1. BƯỚC 1: ROUTING (CHỈ KHI MODE LÀ AUTO)
        intent = "LEGAL"
        if request.mode == ChatMode.AUTO:
            intent = await router.route(request.message, request.history)
        logging.info(f"Query: {request.message} | Intent Detected: {intent} | User Mode: {request.mode}")

        engine = None

        # 2. BƯỚC 2: QUYẾT ĐỊNH ENGINE
        
        # Trường hợp A: Nếu là chuyện phiếm (NON_LEGAL) -> Luôn dùng ChitChat
        if intent == "NON_LEGAL":
            engine = chit_chat_chain
        
        # Trường hợp B: Nếu là câu hỏi pháp luật (LEGAL)
        else:
            # Dựa vào User Mode để chọn nguồn dữ liệu
            
            if request.mode == ChatMode.WEB:
                engine = web_chain
                
            elif request.mode == ChatMode.HYBRID:
                engine = hybrid_chain
                
            else:
                # Bao gồm: ChatMode.AUTO và ChatMode.LAW_DB
                # Mặc định của AUTO là dùng kho văn bản (RAG)
                engine = legal_rag_chain

        # 3. STREAM OR FULL RESPONSE
        if request.stream:
            return StreamingResponse(
                engine.chat(request.message, request.history, rag_engine),
                media_type="application/x-ndjson"
            )
        else:
            # Collect full response
            full_content = ""
            used_docs = []
            
            async for chunk in engine.chat(request.message, request.history, rag_engine):
                try:
                    data = json.loads(chunk)
                    if data['type'] == 'content':
                         if 'delta' in data:
                             full_content += data['delta']
                    elif data['type'] == 'used_docs':
                         if 'data' in data:
                             used_docs = data['data']
                         elif 'ids' in data:
                             used_docs = [{"id": i} for i in data['ids']]
                except:
                    continue
            
            # Remove <think> tags if present in non-streaming mode too
            full_content = re.sub(r'<think>.*?</think>', '', full_content, flags=re.DOTALL).strip()

            return {
                "response": full_content,
                "used_docs": used_docs
            }

    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{doc_id}")
async def get_document(doc_id: str, session: AsyncSession = Depends(get_async_session)):
    """
    Retrieve document detail by ID for UI display.
    Now uses PostgreSQL for fast retrieval instead of Qdrant payload scanning.
    """
    try:
        # Detect source based on ID pattern:
        # - IDs with hyphen (-) are from phapdien (UUID format)
        # - IDs without hyphen are from vbqppl (e.g. "15/2012/TT-BGTVT")
        
        if '-' in doc_id and len(doc_id) == 36:  # UUID pattern for Pháp Điển
            # Query Pháp Điển
            result = await session.execute(
                select(PhapDienDieu).where(PhapDienDieu.id == doc_id)
            )
            dieu = result.scalar_one_or_none()
            
            if dieu:
                # Parse VBQPPL references if available
                refs = []
                if dieu.vbqppl_refs:
                    try:
                        refs = json.loads(dieu.vbqppl_refs)
                    except:
                        pass
                
                return {
                    "metadata": {
                        "id": doc_id,
                        "title": dieu.ten,
                        "url": "https://phapdien.moj.gov.vn/TraCuuPhapDien/MainBoPD.aspx",
                        "source": "phapdien"
                    },
                    "content": [{
                        "type": "text",
                        "title": dieu.ten,
                        "content": dieu.noi_dung
                    }],
                    "references": refs
                }
        
        # Determine if doc_id is a hash (32 hex chars)
        if len(doc_id) == 32 and all(c in '0123456789abcdefABCDEF' for c in doc_id):
            # Find section by hash_id
            section_res = await session.execute(
                select(VBQPPLSection).where(VBQPPLSection.hash_id == doc_id)
            )
            section = section_res.scalar_one_or_none()
            
            if section:
                # Need to fetch parent doc to get title and URL metadata
                doc_res = await session.execute(
                    select(VBQPPLDoc).where(VBQPPLDoc.id == section.doc_id)
                )
                doc = doc_res.scalar_one_or_none()
                doc_title = doc.title if doc else "Unknown Document"
                doc_url = doc.url if doc else "#"

                return {
                    "metadata": {
                        "id": section.doc_id,
                        "title": doc_title,
                        "url": doc_url,
                        "source": "vbqppl"
                    },
                    "content": [{
                        "type": "text",
                        "title": section.hierarchy_path or section.label,
                        "content": section.content
                    }]
                }

        # Query full VBQPPL document by doc_id
        result = await session.execute(
            select(VBQPPLDoc).where(VBQPPLDoc.id == doc_id)
        )
        doc = result.scalar_one_or_none()
        
        if doc:
            # Get all sections for this document
            sections_result = await session.execute(
                select(VBQPPLSection).where(VBQPPLSection.doc_id == doc_id)
            )
            sections = sections_result.scalars().all()
            
            # Build content list from sections
            content_list = []
            for section in sections:
                content_list.append({
                    "type": "text",
                    "title": section.hierarchy_path or section.label,
                    "content": section.content
                })
            
            # If no sections, use full document content
            if not content_list and doc.content:
                content_list = [{
                    "type": "text",
                    "title": doc.title or "Nội dung văn bản",
                    "content": doc.content
                }]
            
            return {
                "metadata": {
                    "id": doc_id, 
                    "title": doc.title or "Unknown",
                    "url": doc.url or "#",
                    "source": "vbqppl"
                },
                "content": content_list,
                "full_content": doc.content
            }
        
        # Fallback: Try Qdrant/Pháp Điển if not found in VBQPPL (legacy support)
        # ... existing fallback code ...
        pass
        
        if not doc:
             # Try Pháp Điển as last resort for non-UUID IDs
             result = await session.execute(
                select(PhapDienDieu).where(PhapDienDieu.id == doc_id)
            )
             dieu = result.scalar_one_or_none()
             
             if dieu:
                 return {
                    "metadata": {
                        "id": doc_id,
                        "title": dieu.ten,
                        "url": "https://phapdien.moj.gov.vn/TraCuuPhapDien/MainBoPD.aspx",
                        "source": "phapdien"
                    },
                    "content": [{
                        "type": "text",
                        "title": dieu.ten,
                        "content": dieu.noi_dung
                    }]
                }
        
        raise HTTPException(status_code=404, detail="Document not found")

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Get Document Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)