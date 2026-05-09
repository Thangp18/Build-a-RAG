import streamlit as st 
import os
from dotenv import load_dotenv
from implementation import ingest_url, ingest_local
from implementation.agent import create_agent_with_retriever, convert_chat_history

load_dotenv(override=True)

def page_setup():
    st.set_page_config(
        page_title="RAG", 
        page_icon="🤖",
        layout="wide"
        )
    st.header(
        "Xây dựng hệ thống RAG",
        divider="rainbow"
        )

def sidebar():
    with st.sidebar:
        st.title("Cấu hình")
        st.header("Nguồn dữ liệu:")
        data_source= st.radio(
            "Chọn nguồn:",
            options= ["🌐 URL", "📁 File từ thiết bị"],
        )
        
        uploaded_files = None
        url = None

        if data_source == "📁 File từ thiết bị":
            uploaded_files= st.file_uploader("Chọn file:", accept_multiple_files=True)
    
        elif data_source == "🌐 URL":
            url= st.text_input("Nhập URL:")

        if st.button("Bắt đầu xử lý"):
            with st.status("Đang xử lý dữ liệu...") as status:
                try:
                    if data_source == "🌐 URL":
                        if not url:
                            st.error("Vui lòng nhập URL.")
                            status.update(label="Lỗi", state="error")
                            return
                        
                        urls = [u.strip() for u in url.split(",") if u.strip()]
                        docs = ingest_url.load_docs(urls)
                        chunks = ingest_url.create_chunks(docs)
                        ensemble_retriever = ingest_url.create_embeddings(chunks)
                        st.session_state.agent_executor = create_agent_with_retriever(ensemble_retriever)
                        
                        status.update(label="Dữ liệu đã sẵn sàng!", state="complete")
                        st.success("Dữ liệu URL đã được xử lý thành công!")

                    elif data_source == "📁 File từ thiết bị":
                        if not uploaded_files:
                            st.error("Vui lòng tải lên ít nhất một file.")
                            status.update(label="Lỗi", state="error")
                            return
                        
                        # Lưu file vào thư mục data
                        data_dir = "./data"
                        os.makedirs(data_dir, exist_ok=True)
                        for file in uploaded_files:
                            file_path = os.path.join(data_dir, file.name)
                            with open(file_path, "wb") as f:
                                f.write(file.getbuffer())
                        
                        docs = ingest_local.load_docs()
                        chunks = ingest_local.create_chunks(docs)
                        ensemble_retriever = ingest_local.create_embeddings(chunks)
                        st.session_state.agent_executor = create_agent_with_retriever(ensemble_retriever)

                        status.update(label="Dữ liệu đã sẵn sàng!", state="complete")
                        st.success("Dữ liệu File đã được xử lý thành công!")

                except Exception as e:
                    status.update(label="Có lỗi xảy ra", state="error")
                    st.error(f"Lỗi: {e}")

def chat_interface():
    # Khởi tạo các biến session_state nếu chưa có
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = None

    # Hiển thị tin nhắn
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Ô nhập chat
    if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
        # 1. Hiển thị câu hỏi
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Gọi LLM
        with st.chat_message("assistant"):
            if st.session_state.agent_executor:
                try:
                    lc_messages = convert_chat_history(st.session_state.messages)
                    response = st.session_state.agent_executor.invoke({
                        "messages": lc_messages
                    })
                    # Kết quả trả về chứa danh sách messages, ta lấy tin nhắn cuối cùng
                    answer = response["messages"][-1].content
                except Exception as e:
                    answer = f"Lỗi khi gọi model: {e}"
            else:
                answer = "Vui lòng nạp dữ liệu ở phần Cấu hình trước khi hỏi." 
            
            st.markdown(answer)
            st.session_state.messages.append({
                "role": "assistant", "content": answer
            })

if __name__ == "__main__":
    page_setup()
    sidebar()
    chat_interface()