# 🤖 Build-a-RAG (Vietnamese RAG Agent)

Dự án này là một ứng dụng **RAG (Retrieval-Augmented Generation)** tiên tiến, được xây dựng bằng Python và Streamlit, kết hợp sức mạnh của tác tử AI (AI Agent) thông qua LangChain và LangGraph để trả lời các câu hỏi dựa trên kho dữ liệu do người dùng cung cấp. Hệ thống được tối ưu đặc biệt cho tiếng Việt.

---

## ✨ Tính năng nổi bật

- 🌐 **Thu thập dữ liệu từ URL**: Tự động trích xuất nội dung từ các trang web (Web Scraping).
- 📁 **Hỗ trợ tải tệp tin cục bộ**: Đọc dữ liệu từ file trên máy tính (hỗ trợ PDF, TXT, DOCX,...) với thuật toán nhận diện nhanh (`unstructured`).
- 🧠 **Cắt câu theo ngữ nghĩa (Semantic Chunking)**: Chia nhỏ tài liệu dựa trên ngữ cảnh để giữ trọn vẹn ý nghĩa của đoạn văn thay vì cắt mù quáng theo độ dài.
- 🔍 **Tìm kiếm lai (Hybrid Search / Ensemble Retriever)**: Kết hợp tìm kiếm theo vector (ChromaDB) và tìm kiếm từ khóa (BM25) để mang lại kết quả truy xuất chính xác nhất.
- 🤖 **ReAct AI Agent**: Ứng dụng không chỉ đơn thuần gọi LLM, mà sử dụng kiến trúc AI Agent (LangGraph) với bộ công cụ `retrieve_context`. Agent có khả năng tự suy luận và tự quyết định lúc nào cần dùng công cụ tìm kiếm trong tài liệu để trả lời người dùng.
- 💬 **Giao diện Chat trực quan**: Giao diện hội thoại Streamlit hiện đại, lưu trữ lịch sử trò chuyện.

---

## 🛠️ Công nghệ sử dụng

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Orchestration**: [LangChain](https://www.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/)
- **LLM**: Google Gemini (`gemini-2.5-flash-lite`) thông qua `langchain-google-genai`
- **Embeddings**: `AITeamVN/Vietnamese_Embedding` (HuggingFace)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Sparse Retriever**: `rank_bm25`

---

## 📂 Cấu trúc dự án

```text
BuildRAG/
├── src/
│   ├── implementation/
│   │   ├── agent.py          # Khởi tạo ReAct Agent và Tool (retrieve_context)
│   │   ├── ingest_local.py   # Xử lý dữ liệu từ file cục bộ
│   │   └── ingest_url.py     # Xử lý dữ liệu từ URL web
│   └── main.py               # Giao diện chính của ứng dụng Streamlit
├── data/                     # Thư mục lưu trữ tạm thời các file được tải lên
├── chroma_db/                # Cơ sở dữ liệu Vector cục bộ
├── .env                      # File chứa biến môi trường (API Key)
├── requirements.txt          # Danh sách thư viện cần thiết
└── README.md
```

---

## 🚀 Hướng dẫn cài đặt

### 1. Yêu cầu hệ thống
- Python 3.9 trở lên
- Git

### 2. Cài đặt chi tiết

Đầu tiên, clone dự án về máy:
```bash
git clone https://github.com/Thangp18/Build-a-RAG.git
cd Build-a-RAG
```

Tạo môi trường ảo (Virtual Environment) và kích hoạt nó:
```bash
# Đối với Windows
python -m venv .venv
.\.venv\Scripts\activate

# Đối với macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

Cài đặt các thư viện phụ thuộc:
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install rank_bm25 "unstructured[pdf]" unstructured-inference
```

### 3. Cấu hình biến môi trường
Tạo một file `.env` ở thư mục gốc của dự án và điền Google API Key của bạn vào:
```env
GOOGLE_API_KEY="AIzaSyYour-Google-Gemini-API-Key-Here"
```
*(Nếu chưa có API Key, bạn có thể tạo miễn phí tại [Google AI Studio](https://aistudio.google.com/)).*

---

## 💻 Hướng dẫn sử dụng

Khởi chạy ứng dụng Streamlit bằng lệnh sau:
```bash
streamlit run src/main.py
```

Trình duyệt sẽ tự động mở tại địa chỉ `http://localhost:8501`. Tại đây bạn có thể:
1. Nhập danh sách URL (ngăn cách bằng dấu phẩy) hoặc tải lên file dữ liệu (PDF, Word, TXT,...).
2. Nhấn nút **"Bắt đầu xử lý"** để hệ thống nhúng (embed) dữ liệu vào bộ nhớ.
3. Bắt đầu chat với AI ở khung bên phải để hỏi các thông tin liên quan đến tài liệu vừa nạp. Hệ thống sẽ trích xuất và giải đáp thắc mắc bằng Tiếng Việt dựa trên nội dung gốc.

---

## 🤝 Đóng góp
Dự án được xây dựng với mục đích học tập và triển khai RAG thực tế. Các đóng góp (Pull requests) và báo lỗi (Issues) luôn được chào đón!

---
*Made with ❤️ by Thangp18.*
