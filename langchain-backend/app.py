from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from chat import ChatRequest, ChatAgent, ChainChat
from rag import RAG

from contextlib import asynccontextmanager
import json
import logging
logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chat_agent, chain_chat, rag_engine
    try:
        chat_agent = ChatAgent()
        chain_chat = ChainChat()
        logging.info("Successfully initialized Chat Strategies (Agent & Chain).")
    except Exception as e:
        logging.error(f"Failed to initialize Chat Strategies: {str(e)}")
        raise

    try:
        rag_engine = RAG()
        logging.info("Successfully initialized RAG.")
    except Exception as e:
        logging.error(f"Failed to initialize RAG: {str(e)}")
        raise
    
    yield
    
    rag_engine.close()
    logging.info("Closed RAG.")


app = FastAPI(
    title="Vietnamese Law RAG Chatbot - LangChain Backend",
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
    return {
        "status": "ok",
        "message": "Law RAG API (LangChain Backend) is running",
        "modes": ["agent", "chain"]
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint.
    - request.mode = 'agent' (default): Uses dynamic tool calling.
    - request.mode = 'chain': Uses deterministic Retrieve-then-Generate.
    """
    if request.mode == "agent":
        engine = chat_agent
    else:
        engine = chain_chat
        
    return StreamingResponse(
        engine.chat(request.message, request.history, rag_engine),
        media_type="application/x-ndjson"
    )

@app.get("/document/{doc_id}")
async def get_document(doc_id: str):
    """
    Retrieve a specific document by ID.
    Searches in the Qdrant collection.
    """
    try:
        # Search for the document in Qdrant by ID
        results = resources.qdrant_client.scroll(
            collection_name=resources.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="id", match=MatchValue(value=doc_id))
                ]
            ),
            limit=1
        )[0]
        
        if not results:
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc = results[0]
        payload = doc.payload
        
        content = payload.get("content", "")
        title = content.split("\n")[0][:100] if content else "Unknown"
        
        return {
            "metadata": {
                "id": payload.get("id", doc_id),
                "title": title,
                "url": payload.get("url", "#")
            },
            "content": [{
                "id": 1,
                "doc_id": payload.get("id", doc_id),
                "anchor": "",
                "type": "article",
                "title": title,
                "content": content,
                "is_structure": False
            }]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
