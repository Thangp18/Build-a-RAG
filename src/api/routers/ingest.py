import os
import shutil
from typing import List
from fastapi import APIRouter, File, UploadFile, HTTPException
from src.api.schemas import URLRequest, ActionResponse
from src.api.agent_state import reset_agent
from src.implementation import ingest_url, ingest_local

router = APIRouter(
    tags=["Ingest"]
)

# 1. Endpoint nạp dữ liệu từ URLs
@router.post("/ingest/url", response_model=ActionResponse)
def ingest_url_endpoint(request: URLRequest):
    try:
        docs= ingest_url.load_docs(request.urls)
        chunks= ingest_url.create_chunks(docs)
        ingest_url.create_embeddings(chunks)

        # Reset agent để nó load lại Tools kết nối tới DB mới nạp
        reset_agent()
        return {"status": "success", "message": f"Xử lý thành công {request.urls} URLs"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 2. Endpoint nạp dữ liệu từ FILE tải lên
@router.post("/ingest/local", response_model=ActionResponse)
def ingest_file_endpoint(files: List[UploadFile] = File(...)):
    try: 
        data_dir = "./data"
        os.makedirs(data_dir, exist_ok=True)
        saved_file_paths = []
        
        for file in files:
            file_path = os.path.join(data_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_file_paths.append(file_path)
            
        # Chạy pipeline xử lý file cục bộ
        docs = ingest_local.load_docs(saved_file_paths)
        chunks = ingest_local.create_chunks(docs)
        ingest_local.create_embeddings(chunks)
        
        # Reset agent để cập nhật database mới
        reset_agent()
        
        return {"status": "success", "message": f"Đã xử lý thành công {len(saved_file_paths)} tệp tin!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))