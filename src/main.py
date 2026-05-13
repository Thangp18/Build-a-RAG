import streamlit as st
import requests

# Địa chỉ của API Backend
API_URL = "http://127.0.0.1:8000"

def page_setup():
    st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")
    st.header("🤖 Trợ lý AI RAG", divider="rainbow")

def sidebar():
    with st.sidebar:
        st.title("⚙️ Cấu hình")
        st.header("Nguồn dữ liệu:")
        data_source = st.radio("Chọn nguồn:", options=["🌐 URL", "📁 File từ thiết bị"])
        
        if data_source == "📁 File từ thiết bị":
            uploaded_files = st.file_uploader("Chọn file:", accept_multiple_files=True)
            if st.button("Bắt đầu xử lý File") and uploaded_files:
                with st.spinner("Backend đang xử lý file..."):
                    # Gửi nhiều file qua form-data
                    files_payload = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
                    try:
                        res = requests.post(f"{API_URL}/ingest/local", files=files_payload)
                        if res.status_code == 200:
                            st.success(res.json()["message"])
                        else:
                            st.error(f"Lỗi: {res.json().get('detail', 'Unknown Error')}")
                    except Exception as e:
                        st.error(f"Không thể kết nối tới API: {e}")
                        
        elif data_source == "🌐 URL":
            url_input = st.text_input("Nhập danh sách URL (ngăn cách bởi dấu phẩy):")
            if st.button("Bắt đầu xử lý URL") and url_input:
                with st.spinner("Backend đang đọc URL..."):
                    urls = [u.strip() for u in url_input.split(",") if u.strip()]
                    try:
                        res = requests.post(f"{API_URL}/ingest/url", json={"urls": urls})
                        if res.status_code == 200:
                            st.success(res.json()["message"])
                        else:
                            st.error(f"Lỗi: {res.json().get('detail', 'Unknown Error')}")
                    except Exception as e:
                        st.error(f"Không thể kết nối tới API: {e}")

def chat_interface():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Hiển thị lịch sử chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Nhập tin nhắn mới
    if prompt := st.chat_input("Nhập câu hỏi của bạn cho tài liệu..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                payload = {"messages": prompt, "session_id": "session_1"}
                # Thêm tham số stream=True để kích hoạt nhận phản hồi dạng streaming
                res = requests.post(f"{API_URL}/chat", json=payload, stream=True)
                
                if res.status_code == 200:
                    # Định nghĩa generator để đọc dữ liệu từ response stream theo từng cụm (chunk)
                    def generate_stream():
                        for chunk in res.iter_content(chunk_size=None, decode_unicode=True):
                            if chunk:
                                yield chunk
                    
                    # Sử dụng st.write_stream để tạo hiệu ứng gõ chữ cực mượt ngay thời gian thực
                    answer = st.write_stream(generate_stream())
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    # Cố gắng đọc lỗi từ backend
                    try:
                        error_msg = res.json().get("detail", "Lỗi không xác định")
                    except:
                        error_msg = res.text
                    st.error(f"Backend báo lỗi: {error_msg}")
            except Exception as e:
                st.error(f"Lỗi kết nối server API: {e}")

if __name__ == "__main__":
    page_setup()
    sidebar()
    chat_interface()
