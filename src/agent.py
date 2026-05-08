from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_milvus import Milvus
from langchain.agents import create_agent
from dotenv import load_dotenv
import os 
load_dotenv(override=True)

if not os.path.exists("./milvus_db"):
    os.makedirs("./milvus_db", exist_ok=True)
DB_NAME = "./milvus_db/milvus.db"
EMBEDDING_MODEL= HuggingFaceEmbeddings(model= "AITeamVN/Vietnamese_Embedding")
vectorstore = Milvus(
    embedding_function=EMBEDDING_MODEL,
    connection_args={"uri": DB_NAME},
    collection_name="vietnamese_rag"
)
model= ChatGoogleGenerativeAI(
    model= "gemini-2.5-flash-lite",
    verbose= True,
    temperature= 0.7,
    google_api_key= os.getenv("GOOGLE_API_KEY"),
    streaming= True
    )

@tool(response_format="content_and_artifact")
def retrieve_context(query):
    retrieved_docs = vectorstore.similarity_search(query, k=3)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized

def agent():
    tools = [retrieve_context]
    system_prompt = (
        "Bạn là trợ lý AI chuyên sâu về các bài viết. "
        "Sử dụng công cụ 'retrieve_context' để truy xuất ngữ cảnh từ các bài viết. "
        "Trả lời câu hỏi của người dùng dựa trên thông tin được truy xuất. "
        "Nếu ngữ cảnh không chứa thông tin liên quan, hãy trả lời: 'Tôi không tìm thấy thông tin liên quan trong các bài viết. ' "
        "Không tự bịa đặt thông tin. Hãy trả lời bằng Tiếng Việt."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_agent(model, tools, system_prompt=prompt)
    return agent