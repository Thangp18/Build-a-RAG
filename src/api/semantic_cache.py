import logging 
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

#Khởi tạo logger
logger = logging.getLogger("semantic_cache")

#Khởi tạo mô hình embeddings và vectorstore
EMBEDDING_MODEL = HuggingFaceEmbeddings(model="AITeamVN/Vietnamese_Embedding")
DB_PATH = "./chroma_db"


class SemanticCache:
    def __init__(self, persist_directory=DB_PATH, threshold=0.2):
        self.threshold = threshold
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=EMBEDDING_MODEL,
            collection_name="semantic_cache"
        )
    
    def lookup(self, query: str):
        """
        Tìm kiếm trong cache dựa trên câu truy vấn đầu vào
        """
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=1)
            if results:
                doc, score = results[0]
                logger.info(f" Kiểm tra Cache: Câu hỏi tương đồng nhất có khoảng cách L2 = {score:.4f}")
                
                # Nếu khoảng cách vector nhỏ hơn ngưỡng threshold -> Hit Cache!
                if score < self.threshold:
                    cached_answer = doc.metadata.get("answer")
                    logger.info(" SEMANTIC CACHE HIT! Tìm thấy câu trả lời trong bộ nhớ đệm.")
                    return cached_answer
            
            logger.info("SEMANTIC CACHE MISS. Không tìm thấy câu trả lời phù hợp.")
            return None
        except Exception as e:
            logger.error(f"Lỗi truy vấn Cache: {e}")
            return None

    def update(self, query: str, answer: str):
        """
        Thêm câu trả lời vào cache
        """
        try:
            self.vectorstore.add_texts(
                texts=[query],
                metadatas=[{"answer": answer}]
            )
            logger.info("💾 Đã lưu câu hỏi và phản hồi mới vào Semantic Cache.")
        except Exception as e:
            logger.error(f"Lỗi cập nhật Cache: {e}")
            