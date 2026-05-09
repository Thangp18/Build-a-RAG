import bs4
import os 
from glob import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_experimental.text_splitter import SemanticChunker

load_dotenv(override=True)

if not os.path.exists("./chroma_db"):
    os.makedirs("./chroma_db", exist_ok=True)
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = HuggingFaceEmbeddings(model="AITeamVN/Vietnamese_Embedding")
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

#Load dữ liệu:
def load_docs(urls):
    loader = WebBaseLoader(
        web_paths=urls
    )
    docs = loader.load()
    
    # Lọc bỏ các document rỗng để tránh lỗi
    valid_docs = [doc for doc in docs if doc.page_content.strip()]
    return valid_docs

#Create chunks:
def create_chunks(docs):
    if not docs:
        return []

    text_splitter = SemanticChunker(
        embeddings=EMBEDDING_MODEL,
        add_start_index=True,
        breakpoint_threshold_amount=0.85
    )
    chunks = text_splitter.split_documents(docs)
    
    # Lọc bỏ các chunk rỗng hoặc chỉ chứa khoảng trắng/kí tự đặc biệt để tránh lỗi chia cho 0 trong BM25
    valid_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    return valid_chunks

#Create embeddings: ensemble retriever hybrid search (Chroma + BM25)
def create_embeddings(chunks):
    if not chunks:
        raise ValueError(
            "Không tìm thấy nội dung hợp lệ từ URL. Vui lòng kiểm tra lại URL hoặc nội dung trang."
        )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=EMBEDDING_MODEL,
        persist_directory=CHROMA_DIR,
        collection_name="rag"
    )
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    bm25_retriever = BM25Retriever.from_documents(chunks, k=3)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )

    return ensemble_retriever


    
