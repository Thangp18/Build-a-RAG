# 🤖 Build-a-RAG

> **Hệ thống RAG (Retrieval-Augmented Generation) tiên tiến với kiến trúc Microservices, được tối ưu cho Tiếng Việt.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-ReAct_Agent-blueviolet)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)](https://www.trychroma.com/)

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Tính năng nổi bật](#-tính-năng-nổi-bật)
- [Công nghệ sử dụng](#️-công-nghệ-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Hướng dẫn cài đặt](#-hướng-dẫn-cài-đặt)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [API Reference](#-api-reference)

---

## 🎯 Giới thiệu

**Build-a-RAG** là một ứng dụng hỏi-đáp thông minh, cho phép người dùng **nạp bất kỳ tài liệu nào** (file PDF, Word, TXT hoặc URL web) và đặt câu hỏi bằng **Tiếng Việt** — hệ thống sẽ trích xuất và trả lời chính xác dựa trên nội dung của tài liệu đó.

Dự án áp dụng kiến trúc **Microservices** tiêu chuẩn ngành: phân tách hoàn toàn Backend (FastAPI) khỏi Frontend (Streamlit), giao tiếp qua REST API với **Streaming Response** thời gian thực.

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND (Streamlit UI)                       │
│              src/main.py  ·  Port 8501                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │  REST API (HTTP)
                           │  ├── POST /chat         (Streaming)
                           │  ├── POST /ingest/url
                           │  └── POST /ingest/local
┌──────────────────────────▼──────────────────────────────────────┐
│                    BACKEND (FastAPI)                             │
│              src/api/main.py  ·  Port 8000                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Routers (src/api/routers/)                  │   │
│  │   chat.py ── ingest.py                                   │   │
│  └────────────────────┬────────────────────────────────────┘   │
│                        │                                         │
│  ┌─────────────────────▼────────────────────────────────────┐  │
│  │           Implementation (src/implementation/)            │  │
│  │  ┌──────────────┐  ┌────────────────┐  ┌─────────────┐  │  │
│  │  │  ingest_url  │  │  ingest_local  │  │    agent    │  │  │
│  │  │   (WebURL)   │  │   (File/PDF)   │  │ (ReAct+RAG) │  │  │
│  │  └──────────────┘  └────────────────┘  └──────┬──────┘  │  │
│  └────────────────────────────────────────────────│─────────┘  │
└───────────────────────────────────────────────────│─────────────┘
                                                    │
              ┌─────────────────────────────────────▼──────┐
              │              ChromaDB (Vector Store)        │
              │              + BM25 (Sparse Retriever)      │
              │           Hybrid Ensemble Retriever         │
              └─────────────────────────────────────────────┘
```

---

## ✨ Tính năng nổi bật

| Tính năng | Mô tả |
|---|---|
| 🌐 **Thu thập từ URL** | Tự động scrape và xử lý nội dung từ trang web bất kỳ |
| 📁 **Upload tài liệu** | Hỗ trợ PDF, DOCX, TXT, PPTX, XLSX và nhiều định dạng khác |
| 🧠 **Semantic Chunking** | Cắt tài liệu theo ngữ nghĩa, giữ nguyên ngữ cảnh thay vì cắt theo độ dài |
| 🔍 **Hybrid Search** | Kết hợp Vector Search (ChromaDB) + Keyword Search (BM25) để tối ưu độ chính xác |
| 🤖 **ReAct AI Agent** | Agent tự suy luận (LangGraph) quyết định khi nào cần tìm kiếm tài liệu |
| ⚡ **Streaming Response** | Câu trả lời xuất hiện token-by-token theo thời gian thực |
| 🔌 **REST API chuẩn** | FastAPI backend với Swagger UI tự động tại `/docs` |
| 🇻🇳 **Tiếng Việt tối ưu** | Sử dụng model embedding `AITeamVN/Vietnamese_Embedding` chuyên biệt |

---

## 🛠️ Công nghệ sử dụng

### Backend
| Thành phần | Công nghệ |
|---|---|
| **API Framework** | [FastAPI](https://fastapi.tiangolo.com/) |
| **AI Orchestration** | [LangChain](https://www.langchain.com/) + [LangGraph](https://langchain-ai.github.io/langgraph/) |
| **LLM** | Google Gemini (`gemini-2.5-flash-lite`) |
| **Embeddings** | `AITeamVN/Vietnamese_Embedding` (HuggingFace) |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) |
| **Sparse Retrieval** | BM25 (`rank_bm25`) |
| **Document Parsing** | [Unstructured](https://unstructured.io/), PyPDF, python-docx |

### Frontend
| Thành phần | Công nghệ |
|---|---|
| **UI Framework** | [Streamlit](https://streamlit.io/) |
| **HTTP Client** | `requests` (với streaming support) |

---

## 📂 Cấu trúc dự án

```
BuildRAG/
├── src/
│   ├── api/                        # 🔌 Backend FastAPI
│   │   ├── main.py                 # Entry point, đăng ký routers
│   │   ├── schemas.py              # Pydantic models (Request/Response)
│   │   ├── agent_state.py          # Quản lý vòng đời Agent (singleton cache)
│   │   └── routers/
│   │       ├── chat.py             # POST /chat (Streaming Response)
│   │       └── ingest.py           # POST /ingest/url | /ingest/local
│   │
│   ├── implementation/             # ⚙️ Core Logic
│   │   ├── agent.py                # ReAct Agent + Hybrid Retriever Tool
│   │   ├── ingest_url.py           # Pipeline: URL → Chunks → ChromaDB
│   │   └── ingest_local.py         # Pipeline: File → Chunks → ChromaDB
│   │
│   └── main.py                     # 🖥️ Frontend Streamlit UI
│
├── chroma_db/                      # Vector database (tự tạo khi chạy)
├── data/                           # Thư mục chứa file tải lên (tự tạo)
├── .env                            # Biến môi trường (API Keys)
├── requirements.txt
└── README.md
```

---

## 🚀 Hướng dẫn cài đặt

### Yêu cầu hệ thống
- **Python** 3.9 trở lên
- **Git**

### Bước 1: Clone dự án

```bash
git clone https://github.com/Thangp18/Build-a-RAG.git
cd Build-a-RAG
```

### Bước 2: Tạo và kích hoạt môi trường ảo

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Bước 3: Cài đặt dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install "fastapi[standard]" rank_bm25
```

### Bước 4: Cấu hình API Key

Tạo file `.env` tại thư mục gốc và điền Google Gemini API Key:

```env
GOOGLE_API_KEY="AIzaSy-Your-API-Key-Here"
```

> 💡 Tạo API Key miễn phí tại [Google AI Studio](https://aistudio.google.com/).

---

## 💻 Hướng dẫn sử dụng

Dự án cần chạy **2 server song song** — Backend API và Frontend UI.

### Terminal 1 — Khởi động Backend API

```bash
uvicorn src.api.main:app --reload
```

Backend sẽ chạy tại `http://127.0.0.1:8000`.  
Truy cập `http://127.0.0.1:8000/docs` để xem Swagger UI đầy đủ.

### Terminal 2 — Khởi động Frontend UI

```bash
streamlit run src/main.py
```

Frontend sẽ chạy tại `http://localhost:8501`.

### Luồng sử dụng

1. **Nạp dữ liệu** (sidebar bên trái):
   - Chọn nguồn dữ liệu: URL hoặc File từ thiết bị
   - Nhập URL hoặc tải lên file và nhấn **"Bắt đầu xử lý"**
   - Hệ thống sẽ parse, chunk, embed và lưu vào ChromaDB

2. **Hỏi đáp** (khung chat chính):
   - Nhập câu hỏi bằng Tiếng Việt
   - AI Agent sẽ tự động tìm kiếm trong tài liệu và phản hồi theo thời gian thực

---

## 📡 API Reference

### `POST /chat`
Gửi câu hỏi cho AI Agent, nhận phản hồi dạng **Streaming**.

```json
// Request Body
{
  "messages": "Tóm tắt nội dung tài liệu cho tôi?",
  "session_id": "session_1"
}
```

```
// Response: text/plain (streamed tokens)
"Dựa trên tài liệu đã nạp, nội dung chính..."
```

---

### `POST /ingest/url`
Nạp dữ liệu từ danh sách URL.

```json
// Request Body
{
  "urls": ["https://example.com/article", "https://example.com/docs"]
}
```

```json
// Response
{
  "status": "success",
  "message": "Xử lý thành công ['https://example.com/...'] URLs"
}
```

---

### `POST /ingest/local`
Nạp dữ liệu từ file tải lên (multipart/form-data).

```
// Form Data
files: [file1.pdf, file2.docx, ...]
```

```json
// Response
{
  "status": "success",
  "message": "Đã xử lý thành công 2 tệp tin!"
}
```

---

## 🤝 Đóng góp

Dự án được xây dựng với mục đích học tập và nghiên cứu RAG trong thực tế. Mọi đóng góp (Pull Request) và báo lỗi (Issue) đều được chào đón!

---

*Made with ❤️ by [Thangp18](https://github.com/Thangp18)*
