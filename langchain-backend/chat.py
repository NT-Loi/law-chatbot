import json
import logging
import re
from typing import List, Optional

import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from tavily import TavilyClient
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from enum import Enum

from prompts import (
    ROUTER_SYSTEM_PROMPT, ROUTER_FEW_SHOT_EXAMPLES,
    SELECT_SYSTEM_PROMPT, SELECT_USER_PROMPT,
    ANSWER_SYSTEM_PROMPT, ANSWER_USER_PROMPT,
    CHIT_CHAT_SYSTEM_PROMPT,
    EXPANSION_SYSTEM_PROMPT, EXPANSION_USER_PROMPT,
    HYBRID_SYSTEM_PROMPT, HYBRID_USER_PROMPT,
    WEB_SEARCH_SYSTEM_PROMPT, WEB_SEARCH_USER_PROMPT,
    REFLECTION_SYSTEM_PROMPT, REFLECTION_USER_PROMPT,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
CHAT_MODEL = os.getenv("CHAT_MODEL", "JunHowie/Qwen3-4B-GPTQ-Int4") 
BASE_URL = os.getenv("URL", "http://localhost:8000/v1")
API_KEY = os.getenv("API_KEY", "EMPTY")
RERANK_THRESHOLD = 0.75

# --- Helper Functions ---
def format_law_docs_for_prompt(docs):
    blocks = []
    for d in docs:
        id = d.get('id', '')
        source = d.get('source', '')
        title = d.get('title', '')
        content = d.get('content', '')
        hierarchy_path = d.get('hierarchy_path', '')
        
        # Thêm dòng phân cách (---) để tách ID khỏi nội dung ngữ nghĩa
        if source == 'vbqppl':
            doc_id = d.get('doc_id', '')
            blocks.append(f"""
            [INTERNAL_ID: {id}]
            TÊN_VĂN_BẢN: Văn bản {doc_id} {title}
            ĐƯỜNG_DẪN: {hierarchy_path}
            NỘI_DUNG: {content}
            --------------------
            """)
        else:
            blocks.append(f"""
            [INTERNAL_ID: {id}]
            TÊN_VĂN_BẢN: {title}
            NỘI_DUNG: {content}
            --------------------
            """)
            
    return "\n".join(blocks)

def parse_selected_ids(llm_output: str) -> List[str]:
    """Extract JSON list from LLM output even if it contains extra text"""
    try:
        # Tìm pattern mảng JSON: ["..."]
        match = re.search(r'\[.*?\]', llm_output, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        return []
    except Exception:
        logging.error(f"Failed to parse IDs from: {llm_output}")
        return []

def clean_reasoning_output(text: str) -> str:
    """
    Loại bỏ nội dung nằm giữa thẻ <think> và </think> (nếu có)
    để lấy kết quả cuối cùng.
    """
    # Pattern tìm thẻ <think>...</think>, cờ re.DOTALL để match cả xuống dòng
    pattern = r"<think>.*?</think>"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)
    return cleaned_text.strip()

# --- Data Models ---
class ChatMode(str, Enum):
    AUTO = "auto"      
    LAW_DB = "law_db"   
    WEB = "web"         
    HYBRID = "hybrid"   

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []
    mode: ChatMode = ChatMode.AUTO 

class ChatRouter:
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
            model=CHAT_MODEL,
            temperature=0.0, # Giữ nhiệt độ thấp nhất để nhất quán
            max_tokens=1024
        )

    async def route(self, query: str, history: List[dict]) -> str:
        # 1. System Instruction
        messages = [SystemMessage(content=ROUTER_SYSTEM_PROMPT)]
        
        # 2. Inject Few-Shot Examples (Nạp ví dụ vào)
        for ex in ROUTER_FEW_SHOT_EXAMPLES:
            if ex["role"] == "user":
                messages.append(HumanMessage(content=ex["content"]))
            else:
                messages.append(AIMessage(content=ex["content"]))
        
        # 3. User Actual Query
        messages.append(HumanMessage(content=query))
        
        # 4. Invoke LLM
        res = await self.llm.ainvoke(messages)
        raw_content = res.content
        intent = clean_reasoning_output(raw_content) 
        
        # Debug log để kiểm tra model trả về gì
        logging.info(f"Router Input: {query} | Raw Output: {intent}")

        # 5. Robust Parsing
        # Đôi khi model trả về "Intent: LEGAL", ta chỉ cần check keyword
        if "NON_LEGAL" in intent or "NON LEGAL" in intent:
            return "NON_LEGAL"
        
        # Nếu không phải NON_LEGAL, mặc định đẩy về LEGAL để an toàn
        return "LEGAL"

class WebSearchEngine:
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            logging.warning("TAVILY_API_KEY not found in .env")
        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str, top_k=50) -> List[dict]:
        """
        Thực hiện search qua Tavily API.
        """
        try:
            logging.info(f"Tavily searching for: {query}")
            
            # Thêm ngữ cảnh "pháp luật Việt Nam" để kết quả chính xác hơn
            # nếu query người dùng quá ngắn
            search_query = query
            if "việt nam" not in query.lower() and "luật" not in query.lower():
                search_query = f"{query} pháp luật Việt Nam"

            # Gọi API Tavily
            # search_depth="advanced": Tìm sâu hơn, chất lượng hơn (tốn 2 credit)
            # search_depth="basic": Nhanh, tiết kiệm (1 credit)
            response = self.client.search(
                query=search_query,
                search_depth="advanced", 
                max_results=top_k,
                include_answer=False, # Chúng ta tự để LLM trả lời
                include_domains=["thuvienphapluat.vn", "luatvietnam.vn", "baochinhphu.vn"] # (Tuỳ chọn) Giới hạn nguồn uy tín
            )

            results = []
            for res in response.get('results', []):
                results.append({
                    "id": res['url'],           # Dùng URL làm ID
                    "title": res['title'],
                    "content": res['content'],  # Tavily tự tóm tắt nội dung quan trọng
                    "url": res['url'],
                    "source_type": "WEB",       # Đánh dấu nguồn
                    "score": res['score']       # Độ liên quan do Tavily tính
                })
            
            return results

        except Exception as e:
            logging.error(f"Tavily Search Error: {str(e)}")
            # Fallback an toàn: Trả về list rỗng để chain không bị crash
            return []

# --- Helper Logic for Streaming with Citations ---
async def stream_with_citations(llm, messages, rag_engine=None):
    """
    Handles streaming response from LLM, parsing <USED_DOCS> tags 
    to separate content from citations.
    """
    buffer = ""
    inside_tag = False
    
    async for chunk in llm.astream(messages):
        content = chunk.content
        if not content:
            continue
        
        buffer += content
        
        # Check for start of tag
        if "<USED_DOCS>" in buffer:
            main_text, remaining = buffer.split("<USED_DOCS>", 1)
            
            if main_text:
                yield json.dumps({"type": "content", "delta": main_text}, ensure_ascii=False) + "\n"
            
            buffer = remaining
            inside_tag = True
            
        elif inside_tag:
            # Inside tag, just accumulate buffer
            pass
            
        else:
            # Normal text handling with safety check for tag start
            if "<" in buffer:
                last_open = buffer.rfind("<")
                potential_tag = buffer[last_open:]
                target = "<USED_DOCS>"
                
                if target.startswith(potential_tag):
                    # Potential partial tag, yield everything before it
                    to_yield = buffer[:last_open]
                    if to_yield:
                        yield json.dumps({"type": "content", "delta": to_yield}, ensure_ascii=False) + "\n"
                    buffer = potential_tag # Keep partial tag in buffer
                else:
                    # Not our tag, yield all
                    yield json.dumps({"type": "content", "delta": buffer}, ensure_ascii=False) + "\n"
                    buffer = ""
            else:
                yield json.dumps({"type": "content", "delta": buffer}, ensure_ascii=False) + "\n"
                buffer = ""

    # End of stream processing
    used_ids = []
    if inside_tag or "<USED_DOCS>" in buffer:
        current_buffer = buffer
        # If we caught the tag start but loop finished
        if "<USED_DOCS>" in current_buffer and not inside_tag:
             _, current_buffer = current_buffer.split("<USED_DOCS>", 1)
        
        # Handle tag closing
        if "</USED_DOCS>" in current_buffer:
            ids_str = current_buffer.split("</USED_DOCS>")[0]
        else:
            ids_str = current_buffer 
            
        clean_ids_str = ids_str.replace(">", "")
        used_ids = [id.strip() for id in clean_ids_str.split(",") if id.strip()]
    else:
        # Flush remaining buffer as text if no tag found
        if buffer:
             yield json.dumps({"type": "content", "delta": buffer}, ensure_ascii=False) + "\n"

    # Send used_docs event
    # Send used_docs event
    if used_ids:
        if rag_engine:
             # Load full details from Qdrant
             rich_docs = await asyncio.to_thread(rag_engine.get_documents_by_ids, used_ids)
             yield json.dumps({"type": "used_docs", "data": rich_docs}, ensure_ascii=False) + "\n"
        else:
             yield json.dumps({"type": "used_docs", "ids": used_ids}, ensure_ascii=False) + "\n"

    logging.info(f"Final Used references: {used_ids}")

class LegalRAGChain:
    def __init__(self):
        # Model cho Selection (Low Temp)
        self.select_llm = ChatOpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
            model=CHAT_MODEL,
            temperature=0.0,
            max_tokens=1024
        )
        # Model cho Answering (Higher Temp)
        self.answer_llm = ChatOpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
            model=CHAT_MODEL,
            temperature=0.2,
            max_tokens=8192,
            streaming=True
        )
        # Model nhanh cho Reflection
        self.llm_fast = ChatOpenAI(
            base_url=BASE_URL, api_key=API_KEY, model=CHAT_MODEL, 
            temperature=0.0, max_tokens=1024 # Tăng token một chút cho suy luận
        )

    async def chat(self, message, history, rag_engine):
        # --- BƯỚC 0: MULTI-QUERY REFLECTION ---
        queries = [message] # Mặc định ít nhất có câu gốc
        
        try:
            reflection_res = await self.llm_fast.ainvoke([
                SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
                HumanMessage(content=REFLECTION_USER_PROMPT.format(question=message))
            ])
            
            # Clean và Parse JSON
            raw_content = clean_reasoning_output(reflection_res.content)
            # Tìm mảng JSON trong text (phòng trường hợp model nói nhảm thêm bên ngoài)
            match = re.search(r'\[.*?\]', raw_content, re.DOTALL)
            if match:
                queries = json.loads(match.group(0))
                logging.info(f"Generated Queries: {queries}")
            else:
                logging.warning("Reflection did not return JSON list, using raw output as single query.")
                queries = [raw_content.strip()]

        except Exception as e:
            logging.error(f"Reflection failed: {e}")
            queries = [message]

         # --- BƯỚC 1: PARALLEL SEARCH (CHỈ SEARCH THÔ) ---
        logging.info(f"Searching Qdrant for {len(queries)} queries parallelly...")
        
        # Mỗi query lấy top 20 thô (chưa rerank)
        tasks = [
            asyncio.to_thread(rag_engine.retrieve, q, top_k=20) 
            for q in queries
        ]
        
        # Chạy song song
        raw_results_list = await asyncio.gather(*tasks)
        
        # --- BƯỚC 2: DEDUPLICATION & MERGE ---
        unique_docs_map = {}
        
        for batch in raw_results_list:
            for doc in batch:
                point_id = doc['id']
                # Nếu doc đã tồn tại, ta có thể giữ lại hoặc cập nhật
                # Ở bước search thô, điểm qdrant_score không so sánh ngang hàng giữa các query khác nhau được
                # nên ta cứ lấy lần xuất hiện đầu tiên hoặc ghi đè đều ổn.
                if point_id not in unique_docs_map:
                    unique_docs_map[point_id] = doc
        
        # Danh sách ứng viên duy nhất để chuẩn bị Rerank
        merged_candidates = list(unique_docs_map.values())
        logging.info(f"Total unique candidates after merge: {len(merged_candidates)}")

        if not merged_candidates:
             yield json.dumps({"type": "error", "content": "Không tìm thấy văn bản liên quan."}, ensure_ascii=False) + "\n"
             return

        # --- BƯỚC 3: SINGLE RERANK (Rerank 1 lần duy nhất) ---
        # QUAN TRỌNG: Rerank dựa trên câu hỏi gốc (message)
        
        ranked_docs = await asyncio.to_thread(
            rag_engine.rerank, 
            query=message,  # <--- Dùng message gốc
            sources=merged_candidates, 
            top_k=10 # Lấy top 10 cuối cùng
        )

        # --- BƯỚC 2: KIỂM TRA ĐỘ TIN CẬY (LOGIC MỚI) ---
        # Kiểm tra điểm của văn bản đầu tiên (văn bản khớp nhất)
        top_score = ranked_docs[0].get("rerank_score", 0)
        filtered_docs = []
        skip_llm_filter = False
        top_k = 5
        if top_score > RERANK_THRESHOLD:
            logging.info(f"High Confidence ({top_score:.4f} > {RERANK_THRESHOLD}). Skipping LLM Selection.")
            
            # Logic: Lấy tất cả các docs có điểm > threshold
            # (Hoặc lấy top_k nếu tất cả đều cao để tránh quá tải context)
            filtered_docs = [d for d in ranked_docs if d.get("rerank_score", 0) > RERANK_THRESHOLD]
            
            # Nếu list quá dài, cắt bớt về top_k để tập trung
            filtered_docs = filtered_docs[:top_k]
            
            skip_llm_filter = True
            
        else:
            logging.info(f"Low Confidence ({top_score:.4f} <= {RERANK_THRESHOLD}). Using LLM Selection.")
            skip_llm_filter = False

        # --- BƯỚC 3: XỬ LÝ LỌC (LLM SELECTION) ---
        if not skip_llm_filter:
            docs_for_selection = ranked_docs[:top_k]
            docs_text_block = format_law_docs_for_prompt(docs_for_selection)
            
            # LƯU Ý: Ở bước lọc này, vẫn nên cho LLM xem câu hỏi gốc (message) 
            # để nó hiểu ngữ cảnh user muốn gì, thay vì câu query khô khan.
            select_messages = [
                SystemMessage(content=SELECT_SYSTEM_PROMPT.format(docs_text=docs_text_block)),
                HumanMessage(content=SELECT_USER_PROMPT.format(question=message)) 
            ]
            
            try:
                selection_response = await self.select_llm.ainvoke(select_messages)
                selected_ids = parse_selected_ids(selection_response.content)
                if selected_ids:
                    filtered_docs = [d for d in ranked_docs if d['id'] in selected_ids]
                else:
                    filtered_docs = ranked_docs[:top_k]
            except Exception as e:
                logging.error(f"LLM Filter Error: {e}")
                filtered_docs = ranked_docs[:top_k]

        # Trả về Client danh sách nguồn
        yield json.dumps({"type": "sources", "data": filtered_docs}, ensure_ascii=False) + "\n"

        if not filtered_docs:
            yield json.dumps({"type": "content", "delta": "Không tìm thấy quy định phù hợp."}, ensure_ascii=False) + "\n"
            return

         # --- BƯỚC 4: ANSWERING ---
        final_context = format_law_docs_for_prompt(filtered_docs)
        
        chat_history_msgs = []
        for h in history[-4:]: 
            if h['role'] == 'user':
                chat_history_msgs.append(HumanMessage(content=h['content']))
            else:
                chat_history_msgs.append(SystemMessage(content=h['content'])) 

        # Dùng câu hỏi gốc (message) để trả lời cho tự nhiên
        answer_messages = [
            SystemMessage(content=ANSWER_SYSTEM_PROMPT.format(context=final_context))
        ] + chat_history_msgs + [HumanMessage(content=ANSWER_USER_PROMPT.format(question=message))]

        async for chunk in stream_with_citations(self.answer_llm, answer_messages, rag_engine=rag_engine):
            yield chunk


class WebLawChain:
    def __init__(self):
        self.web_engine = WebSearchEngine()
        self.llm = ChatOpenAI(
            base_url=BASE_URL, api_key=API_KEY, model=CHAT_MODEL,
            temperature=0.3, streaming=True
        )

    async def chat(self, message, history, rag_engine):
        # Chạy search trong thread pool để không block server
        web_results = await asyncio.to_thread(self.web_engine.search, message, top_k=50)
        
        yield json.dumps({"type": "sources", "data": web_results}, ensure_ascii=False) + "\n"
        
        if not web_results:
            yield json.dumps({"type": "content", "delta": "Không tìm thấy thông tin trên internet."}, ensure_ascii=False) + "\n"
            return
        
        # 2. Return Sources
        yield json.dumps({"type": "sources", "data": web_results}, ensure_ascii=False) + "\n"
        logging.info('Web Results: ', web_results)
        # 3. Answer
        messages = [
            SystemMessage(content=WEB_SEARCH_SYSTEM_PROMPT.format(web_results=json.dumps(web_results, ensure_ascii=False, indent=2))),
            HumanMessage(content=WEB_SEARCH_USER_PROMPT.format(question=message))
        ]
        
        async for chunk in stream_with_citations(self.llm, messages):
            yield chunk

# 3. HybridChain (MỚI: Xử lý cả 2)
class HybridChain:
    def __init__(self):
        self.web_engine = WebSearchEngine()
        self.llm = ChatOpenAI(
            base_url=BASE_URL, api_key=API_KEY, model=CHAT_MODEL,
            temperature=0.2, streaming=True
        )

    async def chat(self, message, history, rag_engine):
        # --- BƯỚC 1: RETRIEVE PARALLEL ---
        
        # Chạy song song: RAG (local/db) và Tavily (network)
        # Vì rag_engine.retrieve hiện tại là sync code (nếu chưa sửa thành async), 
        # ta cũng wrap nó vào to_thread cho chắc chắn
        
        rag_task = asyncio.to_thread(rag_engine.retrieve, message, top_k=50)
        web_task = asyncio.to_thread(self.web_engine.search, message, top_k=50)

        rag_docs, web_docs = await asyncio.gather(rag_task, web_task)

        # Gán nhãn source_type
        for d in rag_docs: d["source_type"] = "LAW_DB"
        for d in web_docs: d["source_type"] = "WEB"

        all_docs = rag_docs + web_docs

        # --- BƯỚC 2: STREAM SOURCES ---
        yield json.dumps({"type": "sources", "data": all_docs}, ensure_ascii=False) + "\n"

        if not all_docs:
             yield json.dumps({"type": "content", "delta": "Không tìm thấy thông tin ở cả kho luật và internet."}, ensure_ascii=False) + "\n"
             return

        # --- BƯỚC 3: FORMAT CONTEXT PHÂN BIỆT NGUỒN ---
        context_blocks = []
        for d in all_docs:
            source_label = "[KHO_LUAT]" if d.get("source_type") == "LAW_DB" else "[INTERNET]"
            context_blocks.append(f"""
            {source_label}
            Tiêu đề: {d.get('title', '')}
            Nội dung: {d['content']}
            """)
        full_context = "\n".join(context_blocks)

        # --- BƯỚC 4: ANSWERING WITH CITATIONS ---
        messages = [
            SystemMessage(content=HYBRID_SYSTEM_PROMPT.format(context=full_context)),
            HumanMessage(content=HYBRID_USER_PROMPT.format(question=message))
        ]

        # Use shared streaming logic
        async for chunk in stream_with_citations(self.llm, messages, rag_engine=rag_engine):
            yield chunk

class ChitChatChain:
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
            model=CHAT_MODEL,
            temperature=0.6,
            streaming=True
        )

    async def chat(self, message, history, rag_engine):
        messages = [SystemMessage(content=CHIT_CHAT_SYSTEM_PROMPT)]
        # Add history
        for h in history[-4:]:
            if h['role'] == 'user':
                messages.append(HumanMessage(content=h['content']))
            else:
                messages.append(SystemMessage(content=h['content'])) # or AIMessage
        
        messages.append(HumanMessage(content=message))

        # ChitChat doesn't return sources
        yield json.dumps({"type": "sources", "data": []}, ensure_ascii=False) + "\n"

        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield json.dumps({"type": "content", "delta": chunk.content}, ensure_ascii=False) + "\n"