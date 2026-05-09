from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# Khởi tạo mô hình ngôn ngữ
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    verbose=True,
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    streaming=True
)

SYSTEM_PROMPT = (
    "Bạn là trợ lý AI chuyên sâu về các bài viết. "
    "Sử dụng công cụ 'retrieve_context' để truy xuất ngữ cảnh từ các bài viết. "
    "Trả lời câu hỏi của người dùng dựa trên thông tin được truy xuất. "
    "Nếu ngữ cảnh không chứa thông tin liên quan, hãy trả lời: 'Tôi không tìm thấy thông tin liên quan trong các bài viết.' "
    "Không tự bịa đặt thông tin. Hãy trả lời bằng Tiếng Việt."
)


def create_agent_with_retriever(ensemble_retriever):
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Truy xuất ngữ cảnh liên quan từ tài liệu đã được lưu trữ."""
        retrieved_docs = ensemble_retriever.invoke(query)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    tools = [retrieve_context]
    agent_executor = create_react_agent(model, tools, prompt=SYSTEM_PROMPT)
    return agent_executor


def convert_chat_history(messages):
    """Chuyển đổi lịch sử chat từ format dict sang LangChain messages."""
    lc_messages = []
    for msg in messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))
    return lc_messages