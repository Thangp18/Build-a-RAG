from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routers import chat, ingest
import logging
import sys

# Thiết lập logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Ghi logs ra console
        logging.FileHandler("api.log")      # Đồng thời lưu logs vào file api.log
    ]
)
logger = logging.getLogger("RAG_API")


app = FastAPI(
    title="Hệ thống AI RAG API",
    description="Backend cho ứng dụng RAG Agent",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức GET, POST, PUT, DELETE
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(ingest.router)

@app.on_event("startup")
def startup():
    logger.info("RAG API đã khởi động thành công!")

@app.get("/")
def root():
    return {"message": "Chào mừng bạn đến với RAG Backend API!"}

@app.on_event("shutdown")
def shutdown():
    logger.info("RAG API đã tắt.")


    
