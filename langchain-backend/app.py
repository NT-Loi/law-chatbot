from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from qdrant_client.models import Filter, FieldCondition, MatchValue
from contextlib import asynccontextmanager
import logging
import json

from chat import ChatRequest, ChatRouter, LegalRAGChain, WebLawChain, ChitChatChain, HybridChain, ChatMode
from rag import RAG

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

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 1. BƯỚC 1: PHÂN LOẠI Ý ĐỊNH (LEGAL vs NON_LEGAL)
        # Luôn luôn phân loại trước để xử lý các câu chào hỏi xã giao
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

        # 3. STREAM RESPONSE
        return StreamingResponse(
            engine.chat(request.message, request.history, rag_engine),
            media_type="application/x-ndjson"
        )

    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{doc_id}")
async def get_document(doc_id: str):
    """
    Retrieve document detail by ID for UI display.
    """
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine not ready")

    try:
        # Helper function for scrolling
        def get_point_by_id(collection, doc_id):
            res, _ = rag_engine.qdrant_client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="id", match=MatchValue(value=doc_id))]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            return res[0] if res else None

        # Try searching in both collections
        point = get_point_by_id(rag_engine.vb_collection_name, doc_id)
        if not point:
            point = get_point_by_id(rag_engine.pd_collection_name, doc_id)
        
        if not point:
             raise HTTPException(status_code=404, detail="Document not found")

        payload = point.payload
        return {
            "metadata": {
                "id": doc_id,
                "title": payload.get("title", "Unknown"),
                "url": payload.get("url", "#"),
                "source": payload.get("source", "unknown")
            },
            "content": [{
                "type": "text",
                "title": payload.get("hierarchy_path", ""),
                "text": payload.get("content", "")
            }]
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Get Document Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)