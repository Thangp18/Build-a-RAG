from langchain_core.tools import create_retriever_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv(override=True)

CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = HuggingFaceEmbeddings(model="AITeamVN/Vietnamese_Embedding")

# Khởi tạo mô hình ngôn ngữ
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    verbose=True,
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    streaming=True
)

def setup_tools():
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        raise FileNotFoundError("Database rỗng hoặc không tồn tại. Vui lòng nạp tài liệu.")
        
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=EMBEDDING_MODEL,
        collection_name="rag"
    )
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    # Lấy toàn bộ bản ghi đã lưu trong ChromaDB
    all_data = vectorstore.get() 
    
    if not all_data or not all_data.get("documents"):
        # Đề phòng trường hợp thư mục tồn tại nhưng rỗng dữ liệu thực tế
        raise FileNotFoundError("Database không chứa văn bản hợp lệ.")
        
    # Đóng gói ngược lại thành List[Document] để nạp cho BM25
    docs = [
        Document(page_content=content, metadata=meta)
        for content, meta in zip(all_data["documents"], all_data["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3
    ensemble_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )
    retriever_tool = create_retriever_tool(
        retriever=ensemble_retriever,
        name="local_docs_search",
        description="Sử dụng công cụ này để tìm kiếm thông tin trong tài liệu nội bộ. Hãy dùng nó mỗi khi người dùng hỏi về thông tin trong tài liệu đã nạp."
    )
    return [retriever_tool]
    
def create_agent(llm, tools):
    SYSTEM_PROMPT = (
        "Bạn là trợ lý AI tên là RagTP. Trả lời bằng Tiếng Việt. "
       "Bạn có quyền sử dụng công cụ {tools} để truy xuất ngữ cảnh từ các bài viết đã nạp. "
        "Hãy trả lời câu hỏi dựa trên thông tin lấy được từ công cụ này. Tránh bịa đặt thông tin."
    )
    # Trả về trực tiếp react agent từ langgraph
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    return agent