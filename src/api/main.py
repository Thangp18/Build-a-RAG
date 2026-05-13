from fastapi import FastAPI
from src.api.routers import chat, ingest

app = FastAPI(
    title="Hệ thống AI RAG API",
    description="Backend cho ứng dụng RAG Agent",
    version="1.0.0"
)

app.include_router(chat.router)
app.include_router(ingest.router)

@app.get("/")
def root():
    return {"message": "Chào mừng bạn đến với RAG Backend API!"}


    
