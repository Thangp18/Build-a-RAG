import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.api.agent_state import get_agent
from dotenv import load_dotenv

load_dotenv()

def main():
    agent = get_agent()

    if agent is None:
        print("Agent rỗng. Nạp tài liệu đi bro")
        return 
    
    test_question = "Dự án BuildRAG sử dụng công nghệ gì cho phần Orchestration?"

    #msg 
    input_msg = {
        "messages": [{
            "role": "user", "content": test_question
        }]
    }

    #call agent
    response= agent.invoke(input=input_msg)

    print("\n🕵️ BẮT ĐẦU PHÂN TÍCH KẾT QUẢ TRẢ VỀ:")
    print("="*50)
    
    extracted_contexts = []
    extracted_answer = ""
    # kiểm tra từng tin nhắn đã suy luận
    for idx, msg in enumerate(response["messages"]):
        # Lấy ra loại Message (Human, AI, hay Tool)
        msg_type = msg.__class__.__name__
        print(f"\n[{idx}] Loại tin nhắn: {msg_type}")

        #Kiểm tra có Agent có dùng tools không 
        if msg_type == "ToolMessage":
            print("TÌM THẤY CONTEXT TỪ TOOL TÌM KIẾM!")
            extracted_contexts.append(msg.content)
            print(f"Nội dung context (nháp): {msg.content[:200]}...")

    
    # kiểm tra answer 
    extracted_answer = response["messages"][-1].content
    print("\n" + "="*50)
    print("🎯 KẾT QUẢ THU HOẠCH ĐƯỢC:")
    print(f"▶️ Answer: {extracted_answer}")
    print(f"▶️ Số lượng Context tìm thấy: {len(extracted_contexts)}")
    
if __name__ == "__main__":
    main()