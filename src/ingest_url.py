import bs4
import os 
from glob import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader, WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores.utils import DistanceStrategy

load_dotenv(override=True)

if not os.path.exists("./milvus_db"):
    os.makedirs("./milvus_db", exist_ok=True)
DB_NAME = "./milvus_db/milvus.db"
EMBEDDING_MODEL= HuggingFaceEmbeddings(model= "AITeamVN/Vietnamese_Embedding")
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

#Load dữ liệu:
def load_docs(urls):
    loader = WebBaseLoader(
        web_paths=urls,
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    return docs

#Create chunks:
def create_chunks(docs):
    text_splitter= SemanticChunker(
        embeddings=EMBEDDING_MODEL,
        add_start_index=True,
        breakpoint_threshold_amount=0.85
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

#Create embeddings: ensemble retriever hybrid search (milvus + BM25)
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


    
