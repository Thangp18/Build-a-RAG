from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.api.schemas import ChatRequest
from src.api.agent_state import get_agent

router = APIRouter(
    tags=["Chat"]
)

# Tạo endpoint Chat hỗ trợ Streaming
@router.post("/chat")
def chat(request: ChatRequest):
    agent = get_agent()
    if agent is None:
        raise HTTPException(
            status_code=400,
            detail="Hệ thống chưa có dữ liệu, Vui lòng nạp dữ liệu trước khi chat!"
        )
    
    def response_generator():
        try:
            input_msg = {"messages": [{
                "role": "user", "content": request.messages
            }]}
            
            # Sử dụng stream_mode="messages" để lấy trực tiếp các token khi LLM đang tạo câu trả lời
            for chunk, metadata in agent.stream(input_msg, stream_mode="messages"):
                # Chỉ lọc lấy các token chữ sinh ra từ mô hình ngôn ngữ (agent node)
                if metadata.get("langgraph_node") == "agent":
                    if hasattr(chunk, "content") and chunk.content:
                        yield chunk.content
        except Exception as e:
            yield f"\n[Lỗi Backend: {str(e)}]"

    # Trả về dữ liệu dưới dạng Stream văn bản thuần (text/plain)
    return StreamingResponse(response_generator(), media_type="text/plain")
