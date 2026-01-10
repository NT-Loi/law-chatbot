from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from pydantic import BaseModel

from rag import RAG

import json
from typing import List, Optional, Any, AsyncGenerator
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []
    mode: str = "agent"  # "agent" or "chain"

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]

chat_model = os.getenv("CHAT_MODEL")
class ChainChat:
    def __init__(self):
        self.llm = init_chat_model(chat_model)
        logging.info(f"Initialized ChainChat with {chat_model}")

    async def chat(self, message: str, history: List[dict], rag_engine: RAG) -> AsyncGenerator[str, None]:
        """
        Classic RAG flow: 
        1. Retrieve Documents (Deterministic)
        2. Build Context
        3. Stream Answer
        """
        try:
            query = message
            sources = rag_engine.retrieve(query)
            # reranked_sources = rag_engine.rerank(query, sources)
            
            yield json.dumps({"type": "sources", "data": sources}, ensure_ascii=False) + "\n"

            docs_text = "\n\n".join([f"Document {i+1}:\n{doc['content']}" for i, doc in enumerate(sources)])
            
            system_prompt = (
                "Bạn là trợ lý pháp luật. Trả lời câu hỏi dựa trên các văn bản pháp luật được cung cấp dưới đây.\n"
                "Nếu thông tin không có trong văn bản, hãy nói là bạn không biết.\n\n"
                f"--- VĂN BẢN PHÁP LUẬT ---\n{docs_text}"
            )
            
            messages = [SystemMessage(content=system_prompt)]
            
            for h in history[-5:]:
                role = h.get("role")
                content = h.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
            
            messages.append(HumanMessage(content=message))

            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    yield json.dumps({"type": "content", "delta": chunk.content}, ensure_ascii=False) + "\n"

        except Exception as e:
            logging.error(f"Error in chain chat: {e}")
            yield json.dumps({"type": "content", "delta": f"Error: {str(e)}"}, ensure_ascii=False) + "\n"


class ChatAgent:
    def __init__(self):
        self.llm = init_chat_model(chat_model)
        logging.info(f"Initialized ChatAgent with {chat_model}")

    async def chat(self, message: str, history: List[dict], rag_engine: RAG) -> AsyncGenerator[str, None]:
        """
        Agentic chat flow that yields NDJSON events.
        Decides dynamically whether to search or just chat.
        """
        @tool
        def search_law(query: str):
            """
            Search for Vietnamese legal documents (Văn bản pháp luật).
            Use this tool whenever the user asks a question about law, regulations, or legal advice.
            """
            logging.info(f"Agent executing tool 'search_law' for query: {query}")
            return rag_engine.retrieve(query)

        tools = [search_law]
        llm_with_tools = self.llm.bind_tools(tools)

        messages = self._build_messages(message, history)
        
        try:
            ai_msg = await llm_with_tools.ainvoke(messages)
            messages.append(ai_msg)

            if ai_msg.tool_calls:
                for tool_call in ai_msg.tool_calls:
                    if tool_call["name"] == "search_law":
                        query_arg = tool_call["args"].get("query", message)
                        sources = rag_engine.retrieve(query_arg)
                        
                        yield json.dumps({"type": "sources", "data": sources}, ensure_ascii=False) + "\n"
                        
                        content_for_llm = json.dumps(sources, ensure_ascii=False)
                        tool_msg = {
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "name": tool_call["name"],
                            "content": content_for_llm
                        }
                        messages.append(tool_msg)

                async for chunk in self.llm.astream(messages):
                    if chunk.content:
                        yield json.dumps({"type": "content", "delta": chunk.content}, ensure_ascii=False) + "\n"
            
            else:
                yield json.dumps({"type": "content", "delta": ai_msg.content}, ensure_ascii=False) + "\n"

        except Exception as e:
            logging.error(f"Error in chat agent: {e}")
            yield json.dumps({"type": "content", "delta": f"Error: {str(e)}"}, ensure_ascii=False) + "\n"

    def _build_messages(self, query: str, history: List[dict]) -> List[BaseMessage]:
        system_prompt = (
            "Bạn là trợ lý pháp luật AI chuyên nghiệp. "
            "Bạn có quyền truy cập vào cơ sở dữ liệu pháp luật Việt Nam thông qua công cụ `search_law`. "
            "QUY TẮC:\n"
            "1. Nếu câu hỏi liên quan đến pháp luật, BẮT BUỘC dùng `search_law`.\n"
            "2. Trả lời dựa trên thông tin tìm thấy.\n"
            "3. Nếu không tìm thấy thông tin, hãy nói rõ."
        )
        msgs = [SystemMessage(content=system_prompt)]
        
        for h in history[-5:]:
            role = h.get("role")
            content = h.get("content", "")
            if role == "user":
                msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                msgs.append(AIMessage(content=content))
        
        msgs.append(HumanMessage(content=query))
        return msgs