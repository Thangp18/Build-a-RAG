from langchain_core.tools import create_retriever_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openrouter import ChatOpenRouter
from langchain_openai import OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank
from dotenv import load_dotenv
import os

load_dotenv(override=True)

CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = HuggingFaceEmbeddings(model="AITeamVN/Vietnamese_Embedding")



# Khởi tạo mô hình ngôn ngữ
llm = ChatOpenRouter(
    model="openrouter/owl-alpha",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
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
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    # Lấy toàn bộ bản ghi đã lưu trong ChromaDB
    all_data = vectorstore.get() 
    
    if not all_data or not all_data.get("documents"):
        raise FileNotFoundError("Database không chứa văn bản hợp lệ.")
        
    # Đóng gói để nạp cho BM25
    docs = [
        Document(page_content=content, metadata=meta)
        for content, meta in zip(all_data["documents"], all_data["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 10
    ensemble_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )

    #Rerank
    reranker = FlashrankRerank(top_n=5)

    #Kết hợp Base Retriever + Reranker
    rerank_retriever = ContextualCompressionRetriever(
        base_retriever= ensemble_retriever,
        base_compressor= reranker
    )
    retriever_tool = create_retriever_tool(
        retriever=rerank_retriever,
        name="local_docs_search",
        description="Sử dụng công cụ này để tìm kiếm thông tin trong tài liệu nội bộ. Hãy dùng nó mỗi khi người dùng hỏi về thông tin trong tài liệu đã nạp."
    )
    return [retriever_tool]

def create_agent(llm, tools):
    SYSTEM_PROMPT = (
        "Bạn là trợ lý AI tên là RagTP. Trả lời bằng Tiếng Việt.\n"
        "Khi nhận được câu hỏi của người dùng, hãy phân tích lịch sử trò chuyện để xác định xem người dùng đang ám chỉ điều gì.\n"
        "Nếu cần truy xuất tài liệu, hãy tạo ra một CÂU LỆNH TRUY VẤN tìm kiếm chứa đầy đủ các từ khóa ngữ cảnh (Query Expansion) "
        "để đưa vào công cụ 'local_docs_search'.\n"
        "Hãy trả lời câu hỏi dựa trên thông tin lấy được từ công cụ này. Tránh bịa đặt thông tin."
    )
    # Tạo ReAct Agent
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    return agent

if __name__ == "__main__":
    tools = setup_tools()
    agent = create_agent(llm, tools)
    print(agent.invoke({"messages": [{"role": "user", "content": "Tôi muốn tìm hiểu về cách sử dụng công cụ local_docs_search"}]}))