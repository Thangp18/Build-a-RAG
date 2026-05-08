import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_experimental.text_splitter import SemanticChunker


load_dotenv(override=True)
DATA_PATH= "../data"
if not os.path.exists("./milvus_db"):
    os.makedirs("./milvus_db", exist_ok=True)
DB_NAME = "./milvus_db/milvus.db"

EMBEDDING_MODEL= HuggingFaceEmbeddings(model= "AITeamVN/Vietnamese_Embedding")

def load_docs():
    #Load dữ liệu:
    loader = DirectoryLoader(
        path= DATA_PATH, 
        glob="**/*",
        loader_cls= UnstructuredFileLoader,
        show_progress=True,
        use_multithreading=True
    )
    docs = loader.load()
    return docs

def create_chunks(docs):
    text_splitter= SemanticChunker(
        embeddings=EMBEDDING_MODEL,
        add_start_index=True,
        breakpoint_threshold_amount=0.85
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

def create_embeddings(chunks):
    vectorstore = Milvus.from_documents(
        documents=chunks,
        embedding=EMBEDDING_MODEL,
        connection_args={"uri": DB_NAME},
        drop_old=True
    )
    milvus_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    bm25_retriever = BM25Retriever.from_documents(chunks,k=3)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[milvus_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )

    return ensemble_retriever


    
