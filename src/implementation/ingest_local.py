import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader, PyPDFLoader
# from langchain_openai import OpenAIEmbeddings
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
# EMBEDDING_MODEL = OpenAIEmbeddings(
#     model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
#     openai_api_base="https://openrouter.ai/api/v1",
#     openai_api_key=os.getenv("OPENROUTER_API_KEY"),
# )


def load_docs(file_paths):
    docs = []
    for path in file_paths:
        ext = os.path.splitext(path)[-1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path=path)
            else:
                loader = UnstructuredFileLoader(
                    file_path=path,
                    strategy="fast",
                )
            docs.extend(loader.load())
        except Exception as e:
            print(f"Lỗi khi load file {path}: {e}")
            # Fallback nếu PyPDFLoader lỗi: Thử lại bằng Unstructured
            try:
                loader = UnstructuredFileLoader(file_path=path)
                docs.extend(loader.load())
            except:
                pass
    
    # Lọc bỏ các document rỗng để tránh lỗi
    valid_docs = [doc for doc in docs if doc.page_content.strip()]
    return valid_docs

def create_chunks(docs):
    if not docs:
        return []

    text_splitter = SemanticChunker(
        embeddings=EMBEDDING_MODEL,
        add_start_index=True,
        breakpoint_threshold_amount=0.85
    )
    chunks = text_splitter.split_documents(docs)
    
    # Lọc bỏ các chunk rỗng để tránh lỗi chia cho 0 trong BM25
    valid_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    return valid_chunks

def create_embeddings(chunks):
    if not chunks:
        raise ValueError(
            "Không tìm thấy nội dung hợp lệ từ file. Vui lòng tải lên ít nhất một file có nội dung."
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


    
