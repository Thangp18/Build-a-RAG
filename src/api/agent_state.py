import sys
import os

# Thêm root project vào path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.implementation.agent import create_agent, setup_tools, llm

# Biến toàn cục lưu Agent
agent_executor = None

def get_agent():
    global agent_executor
    if agent_executor is None:
        try:
            tools = setup_tools()
            agent_executor = create_agent(llm, tools)
            print("🔄 Khởi tạo Agent thành công từ Database hiện tại!")
        except Exception as e:
            print(f"⚠️ Chưa thể khởi tạo Agent (Có thể DB trống): {e}")
    return agent_executor

def reset_agent():
    """Hàm dùng để xoá cache agent mỗi khi nạp dữ liệu mới"""
    global agent_executor
    agent_executor = None
    print("🔄 Đã reset Cache Agent để cập nhật Database mới.")
