from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    messages: str = Field(..., description="Câu hỏi cần truy vấn") 
    session_id: str = Field("default", description="ID của phiên chat")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Câu trả lời của AI")

class URLRequest(BaseModel):
    urls: List[str] = Field(..., description="Danh sách URL cần nạp")

class ActionResponse(BaseModel):
    message: str
    status: str
