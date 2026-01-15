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
    HYBRID_SYSTEM_PROMPT, HYBRID_USER_PROMPT
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
        doc_id = d.get('id', '')
        source = d.get('source', '')
        title = d.get('title', '')
        content = d.get('content', '')
        hierarchy_path = d.get('hierarchy_path', '')
        
        if source == 'vbqppl':
            display_id = d.get('original_doc_id', doc_id) 
            
            blocks.append(f"""
                [DOC_ID: {doc_id} | Nguồn: Văn bản quy phạm pháp luật]
                Tiêu đề: {display_id} {title}
                Đường dẫn: {hierarchy_path}
                Nội dung: {content}
                """)
        else:
            blocks.append(f"""
                [DOC_ID: {doc_id} | Nguồn: Pháp điển]
                Tiêu đề: {title}
                Nội dung: {content}
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

        # --- FIX Ở ĐÂY ---
        raw_content = res.content
        intent = clean_reasoning_output(raw_content) 
        # -----------------
        
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

    def search(self, query: str, top_k=5) -> List[dict]:
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

# --- Chains ---
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
        # Model nhanh cho expansion (như đã thêm ở bước trước)
        self.llm_fast = ChatOpenAI(
            base_url=BASE_URL, api_key=API_KEY, model=CHAT_MODEL, 
            temperature=0.0, max_tokens=1024
        )

    async def chat(self, message, history, rag_engine):
        # --- BƯỚC 0: QUERY EXPANSION (Giữ nguyên) ---
        try:
            expansion_res = await self.llm_fast.ainvoke([
                SystemMessage(content=EXPANSION_SYSTEM_PROMPT),
                HumanMessage(content=EXPANSION_USER_PROMPT.format(question=message))
            ])
            # --- FIX Ở ĐÂY ---
            raw_content = expansion_res.content
            keywords = clean_reasoning_output(raw_content) 
            # -----------------

            logging.info(f"Query: {message} | Expanded Keywords: {keywords}")
            
            # Nếu model trả về rỗng sau khi clean (trường hợp lỗi), dùng query gốc
            if not keywords:
                keywords = message

            expanded_query = f"{message}\nCác thuật ngữ liên quan: {keywords}"
        except Exception:
            expanded_query = message

        # --- BƯỚC 1: RETRIEVAL ---
        # Lấy top_k lớn (ví dụ 10) để có không gian cho LLM lọc nếu cần
        raw_docs = rag_engine.retrieve(expanded_query, top_k=10)
        
        if not raw_docs:
             yield json.dumps({"type": "error", "content": "Không tìm thấy văn bản liên quan."}, ensure_ascii=False) + "\n"
             return

        # --- BƯỚC 2: KIỂM TRA ĐỘ TIN CẬY (LOGIC MỚI) ---
        
        # Kiểm tra điểm của văn bản đầu tiên (văn bản khớp nhất)
        top_score = raw_docs[0].get("score", 0)
        filtered_docs = []
        skip_llm_filter = False

        if top_score > RERANK_THRESHOLD:
            logging.info(f"High Confidence ({top_score:.4f} > {RERANK_THRESHOLD}). Skipping LLM Selection.")
            
            # Logic: Lấy tất cả các docs có điểm > threshold
            # (Hoặc lấy top 5 nếu tất cả đều cao để tránh quá tải context)
            filtered_docs = [d for d in raw_docs if d.get("rerank_score", 0) > RERANK_THRESHOLD]
            
            # Nếu list quá dài, cắt bớt về top 5 để tập trung
            filtered_docs = filtered_docs[:5]
            
            skip_llm_filter = True
            
        else:
            logging.info(f"Low Confidence ({top_score:.4f} <= {RERANK_THRESHOLD}). Using LLM Selection.")
            skip_llm_filter = False

        # --- BƯỚC 3: XỬ LÝ LỌC (TÙY ĐIỀU KIỆN) ---
        
        if not skip_llm_filter:
            # === CHẠY LLM FILTER (Logic cũ) ===
            docs_text_block = format_law_docs_for_prompt(raw_docs)
            select_messages = [
                SystemMessage(content=SELECT_SYSTEM_PROMPT.format(docs_text=docs_text_block)),
                HumanMessage(content=SELECT_USER_PROMPT.format(question=message))
            ]
            
            try:
                selection_response = await self.select_llm.ainvoke(select_messages)
                selected_ids = parse_selected_ids(selection_response.content)
                if selected_ids:
                    filtered_docs = [d for d in raw_docs if d['id'] in selected_ids]
                else:
                    # Fallback: Nếu LLM lọc ra rỗng nhưng Rerank có kết quả, lấy tạm top 1-2
                    filtered_docs = raw_docs[:2]
            except Exception as e:
                logging.error(f"LLM Filter Error: {e}")
                filtered_docs = raw_docs[:3] # Fallback an toàn

        # Trả về Client danh sách nguồn đã chốt
        yield json.dumps({"type": "sources", "data": filtered_docs}, ensure_ascii=False) + "\n"

        if not filtered_docs:
            yield json.dumps({"type": "content", "delta": "Không tìm thấy quy định phù hợp."}, ensure_ascii=False) + "\n"
            return

         # --- BƯỚC 4: ANSWERING VỚI CITATION EXTRACTION ---
        final_context = format_law_docs_for_prompt(filtered_docs)
        
        chat_history_msgs = []
        for h in history[-4:]: # Last 2 turns
            if h['role'] == 'user':
                chat_history_msgs.append(HumanMessage(content=h['content']))
            else:
                chat_history_msgs.append(SystemMessage(content=h['content'])) # or AIMessage

        answer_messages = [
            SystemMessage(content=ANSWER_SYSTEM_PROMPT.format(context=final_context))
        ] + chat_history_msgs + [HumanMessage(content=ANSWER_USER_PROMPT.format(question=message))]

        # Buffer để chứa các mảnh text đang chờ xử lý (tránh in thẻ <USED_DOCS> ra màn hình)
        buffer = ""
        inside_tag = False
        
        async for chunk in self.answer_llm.astream(answer_messages):
            content = chunk.content
            if not content:
                continue
            
            buffer += content
            
            # Kiểm tra xem thẻ mở <USED_DOCS> có bắt đầu xuất hiện không
            if "<USED_DOCS>" in buffer:
                # Tách phần nội dung trả lời và phần ID
                main_text, remaining = buffer.split("<USED_DOCS>", 1)
                
                # 1. Đẩy nốt phần text còn lại cho client
                if main_text:
                    yield json.dumps({"type": "content", "delta": main_text}, ensure_ascii=False) + "\n"
                
                # 2. Chuyển sang chế độ xử lý ID (không in ra content nữa)
                buffer = remaining # buffer giờ chỉ chứa nội dung trong thẻ
                inside_tag = True
                
            elif inside_tag:
                # Đang ở trong thẻ, chỉ gom buffer để parse sau, không yield content
                pass
                
            else:
                # Logic buffer an toàn: Chỉ yield khi chắc chắn không phải là một phần của thẻ <USED_DOCS>
                # Ví dụ: buffer = "Xin chào <US" -> Chưa yield vội, đợi chunk sau ghép vào
                # Cách đơn giản: Yield tất cả trừ phần cuối nếu nó giống thẻ mở
                
                # Để đơn giản hóa cho demo: Ta yield luôn nếu chưa thấy dấu hiệu thẻ
                # (Trong môi trường production cần logic "Sliding Window" kỹ hơn)
                if "<" in buffer:
                    # Có thể là bắt đầu thẻ, giữ lại xử lý sau
                    pass 
                else:
                    yield json.dumps({"type": "content", "delta": buffer}, ensure_ascii=False) + "\n"
                    buffer = ""

        # --- KẾT THÚC STREAM ---
        # Lúc này buffer chứa nội dung trong thẻ (id1, id2...</USED_DOCS>) hoặc text còn dư
        
        used_ids = []
        if inside_tag or "<USED_DOCS>" in buffer:
            # Xử lý nội dung trong thẻ (đã tách phần đầu <USED_DOCS> nếu inside_tag=True)
            # Nếu chưa tách (trường hợp thẻ nằm trọn trong chunk cuối), strip nó đi
            current_buffer = buffer
            if "<USED_DOCS>" in current_buffer and not inside_tag:
                 _, current_buffer = current_buffer.split("<USED_DOCS>", 1)
            
            # Cắt đúng tại thẻ đóng </USED_DOCS>
            if "</USED_DOCS>" in current_buffer:
                ids_str = current_buffer.split("</USED_DOCS>")[0]
            else:
                ids_str = current_buffer # Trường hợp model dừng giữa chừng
                
            # Clean các ký tự còn sót (ví dụ > nếu model output lỗi)
            clean_ids_str = ids_str.replace(">", "")
            
            # Tách ID
            used_ids = [id.strip() for id in clean_ids_str.split(",") if id.strip()]
        else:
            # Trường hợp buffer còn sót text thường (LLM quên output thẻ)
            if buffer:
                 yield json.dumps({"type": "content", "delta": buffer}, ensure_ascii=False) + "\n"

        # Gửi sự kiện riêng chứa list ID đã dùng cho Frontend highlight
        if used_ids:
             yield json.dumps({"type": "used_docs", "ids": used_ids}, ensure_ascii=False) + "\n"

        logging.info(f"Used references: {used_ids}")

class WebLawChain:
    def __init__(self):
        self.web_engine = WebSearchEngine()
        self.llm = ChatOpenAI(
            base_url=BASE_URL, api_key=API_KEY, model=CHAT_MODEL,
            temperature=0.3, streaming=True
        )

    async def chat(self, message, history, rag_engine):
        # Chạy search trong thread pool để không block server
        web_results = await asyncio.to_thread(self.web_engine.search, message, top_k=5)
        
        yield json.dumps({"type": "sources", "data": web_results}, ensure_ascii=False) + "\n"
        
        if not web_results:
            yield json.dumps({"type": "content", "delta": "Không tìm thấy thông tin trên internet."}, ensure_ascii=False) + "\n"
            return
        
        # 2. Return Sources
        yield json.dumps({"type": "sources", "data": web_results}, ensure_ascii=False) + "\n"
        
        # 3. Answer
        context = format_law_docs_for_prompt(web_results) # Tái sử dụng hàm format
        messages = [
            SystemMessage(content=f"Bạn là trợ lý tra cứu tin tức pháp luật. Trả lời dựa trên context sau:\n{context}"),
            HumanMessage(content=message)
        ]
        
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield json.dumps({"type": "content", "delta": chunk.content}, ensure_ascii=False) + "\n"

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
        
        rag_task = asyncio.to_thread(rag_engine.retrieve, message, top_k=5)
        web_task = asyncio.to_thread(self.web_engine.search, message, top_k=3)

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

        # --- BƯỚC 4: ANSWERING ---
        messages = [
            SystemMessage(content=HYBRID_SYSTEM_PROMPT.format(context=full_context)),
            HumanMessage(content=HYBRID_USER_PROMPT.format(question=message))
        ]

        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield json.dumps({"type": "content", "delta": chunk.content}, ensure_ascii=False) + "\n"

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