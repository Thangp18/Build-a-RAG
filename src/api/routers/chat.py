from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.api.schemas import ChatRequest
from src.api.agent_state import get_agent
from src.api.semantic_cache import SemanticCache
import logging
import time
import json

#Khởi tạo logger và cache
logger = logging.getLogger("chat_router")
cache = SemanticCache(
    persist_directory="./chroma_cache",
    threshold=0.2
)

router = APIRouter(
    tags=["Chat"]
)

# Tạo endpoint Chat hỗ trợ Streaming
@router.post("/chat")
def chat(request: ChatRequest):
    start_time = time.time()
    user_query = request.messages
    # kiểm tra cache
    cached_answer = cache.lookup(user_query)
    if cached_answer:
        process_time = round(time.time() - start_time, 3)
        logger.info(f"Trả lời từ Cache cực nhanh trong {process_time} giây.")
        
        def cache_generator():
            # Trả về dạng streaming NDJSON tương thích
            yield json.dumps({"type": "token", "content": f"[Cached] {cached_answer}"}) + "\n"
            yield json.dumps({"type": "docs", "content": "📌 *Thông tin được trả về từ Semantic Cache (không cần truy xuất lại cơ sở dữ liệu).*"}) + "\n"
            
        return StreamingResponse(cache_generator(), media_type="application/x-ndjson")

    # gọi agent xử lý
    agent = get_agent()
    if agent is None:
        logger.error("Không có agent hoặc cơ sở dữ liệu trống.")
        raise HTTPException(
            status_code=400,
            detail="Hệ thống chưa có dữ liệu, Vui lòng nạp dữ liệu trước khi chat!"
        )
    
    def response_generator():
        try:
            input_msg = {"messages": [{
                "role": "user", "content": user_query
            }]}
            
            full_answer = []
            retrieved_docs = []
            
            # Sử dụng stream_mode="messages" để lấy trực tiếp các token khi LLM đang tạo câu trả lời
            for chunk, metadata in agent.stream(input_msg, stream_mode="messages"):
                node = metadata.get("langgraph_node")
                # Chỉ lọc lấy các token chữ sinh ra từ mô hình ngôn ngữ (agent node)
                if node == "agent":
                    if hasattr(chunk, "content") and chunk.content:
                        text = chunk.content
                        full_answer.append(text)
                        yield json.dumps({"type": "token", "content": text}) + "\n"
                # Lấy các tài liệu liên quan được trả về từ tool retriever
                elif node == "tools":
                    if hasattr(chunk, "content") and chunk.content:
                        retrieved_docs.append(chunk.content)
            
            # tính toán thời gian 
            process_time = round(time.time() - start_time, 3)
            logger.info(f"Trả lời từ Agent trong {process_time} giây.")

            # Ghép toàn bộ câu trả lời để lưu cache
            final_answer = "".join(full_answer)
            if final_answer:
                # Sửa lỗi lưu user_query: lưu đúng câu trả lời thực tế của AI
                cache.update(user_query, final_answer)
                logger.info("Đã lưu câu trả lời chính thức vào Semantic Cache.")

            # Gửi các tài liệu thu thập được tới client qua stream
            if retrieved_docs:
                docs_text = "\n\n---\n\n".join(retrieved_docs)
                yield json.dumps({"type": "docs", "content": docs_text}) + "\n"
            else:
                yield json.dumps({"type": "docs", "content": "ℹ️ *Không tìm thấy tài liệu nào có độ liên quan trực tiếp tới câu trả lời.*"}) + "\n"

        except Exception as e:
            yield json.dumps({"type": "token", "content": f"\n[Lỗi Backend: {str(e)}]"}) + "\n"

    # Trả về dữ liệu dưới dạng NDJSON stream
    return StreamingResponse(response_generator(), media_type="application/x-ndjson")
