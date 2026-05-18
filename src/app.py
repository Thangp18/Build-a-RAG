import streamlit as st
import requests
import json

# Địa chỉ của API Backend
API_URL = "http://127.0.0.1:8000"

def page_setup():
    st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")
    st.header("🤖 Trợ lý AI RAG", divider="rainbow")

def sidebar():
    # Khởi tạo danh sách URL trong session state
    if "url_list" not in st.session_state:
        st.session_state.url_list = [""]
    if "url_statuses" not in st.session_state:
        st.session_state.url_statuses = []

    with st.sidebar:
        st.title("⚙️ Cấu hình")
        st.header("Nguồn dữ liệu:")
        data_source = st.radio("Chọn nguồn:", options=["🌐 URL", "📁 File từ thiết bị"])
        
        if data_source == "📁 File từ thiết bị":
            uploaded_files = st.file_uploader("Chọn file:", accept_multiple_files=True)
            if st.button("Bắt đầu xử lý File") and uploaded_files:
                with st.spinner("Đang xử lý file..."):
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
            st.markdown("**Danh sách URL cần xử lý:**")

            # Render từng ô nhập URL với nút xóa
            for i in range(len(st.session_state.url_list)):
                col_input, col_del = st.columns([5, 1])
                with col_input:
                    st.session_state.url_list[i] = st.text_input(
                        f"URL {i + 1}",
                        value=st.session_state.url_list[i],
                        key=f"url_input_{i}",
                        placeholder="https://example.com/page",
                        label_visibility="collapsed",
                    )
                with col_del:
                    if st.button("✕", key=f"del_url_{i}", help="Xóa URL này"):
                        st.session_state.url_list.pop(i)
                        # Xóa trạng thái cũ khi danh sách thay đổi
                        st.session_state.url_statuses = []
                        st.rerun()

            # Nút thêm URL mới
            if st.button("➕ Thêm URL"):
                st.session_state.url_list.append("")
                st.session_state.url_statuses = []
                st.rerun()

            st.divider()

            # Nút xử lý tất cả URL
            if st.button("🚀 Bắt đầu xử lý URL", type="primary"):
                urls = [u.strip() for u in st.session_state.url_list if u.strip()]
                if not urls:
                    st.warning("⚠️ Vui lòng nhập ít nhất một URL.")
                else:
                    statuses = []
                    progress = st.progress(0, text="Đang xử lý...")
                    for idx, url in enumerate(urls):
                        progress.progress(
                            (idx + 1) / len(urls),
                            text=f"Đang xử lý [{idx + 1}/{len(urls)}]: {url[:50]}..."
                        )
                        try:
                            res = requests.post(
                                f"{API_URL}/ingest/url",
                                json={"urls": [url]},
                                timeout=60
                            )
                            if res.status_code == 200:
                                statuses.append({"url": url, "ok": True, "msg": res.json().get("message", "Thành công")})
                            else:
                                statuses.append({"url": url, "ok": False, "msg": res.json().get("detail", "Lỗi không xác định")})
                        except Exception as e:
                            statuses.append({"url": url, "ok": False, "msg": str(e)})
                    progress.empty()
                    st.session_state.url_statuses = statuses
                    st.rerun()

            # Hiển thị kết quả xử lý từng URL
            if st.session_state.url_statuses:
                st.markdown("**Kết quả xử lý:**")
                ok_count = sum(1 for s in st.session_state.url_statuses if s["ok"])
                total = len(st.session_state.url_statuses)
                st.caption(f"✅ {ok_count}/{total} URL xử lý thành công")
                for status in st.session_state.url_statuses:
                    label = status["url"][:40] + "..." if len(status["url"]) > 40 else status["url"]
                    if status["ok"]:
                        st.success(f"✅ {label}")
                    else:
                        st.error(f"❌ {label}\n{status['msg']}")

def chat_interface():
    # Khởi tạo trạng thái nếu chưa có
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_docs" not in st.session_state:
        st.session_state.last_docs = ""

    # Tạo cấu trúc 2 cột: Cột chính (Chat) và Cột phụ bên phải (Tài liệu tham khảo)
    col_chat, col_docs = st.columns([2.5, 1], gap="large")

    with col_docs:
        st.subheader("📄 Tài liệu tham khảo")
        # Tạo placeholder cố định giúp hiển thị tài liệu tức thì
        docs_placeholder = st.empty()
        
        # Render dữ liệu mặc định hoặc trước đó
        with docs_placeholder.container():
            if st.session_state.last_docs:
                st.markdown(
                    f'<div style="background-color: #f1f3f4; color: #202124; padding: 15px; border-radius: 8px; '
                    f'border-left: 5px solid #0078D4; font-size: 0.95rem; line-height: 1.6; '
                    f'max-height: 75vh; overflow-y: auto;">'
                    f'{st.session_state.last_docs.replace(chr(10), "<br>")}'
                    f'</div>', 
                    unsafe_allow_html=True
                )
            else:
                st.info("ℹ️ Các tài liệu và ngữ cảnh được RAG truy xuất sẽ xuất hiện tại đây sau câu trả lời của AI.")

    with col_chat:
        st.subheader("💬 Trò chuyện")
        
        # Dùng container để gom nhóm TOÀN BỘ tin nhắn, đảm bảo hộp nhập liệu sẽ luôn nằm phía dưới cùng
        chat_container = st.container()
        
        with chat_container:
            # Hiển thị lịch sử chat đã lưu trước đó
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    # Vùng nhận tin nhắn mới của người dùng - ĐẶT Ở NGOÀI CÙNG ĐỂ STREAMLIT NEO Ở DƯỚI TRANG
    if prompt := st.chat_input("Nhập câu hỏi của bạn cho tài liệu..."):
        # Thêm ngay tin nhắn User vào session_state và vẽ vào Container phía trên
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            # Tiến hành sinh và vẽ tin nhắn Assistant trong container
            with st.chat_message("assistant"):
                try:
                    payload = {"messages": prompt, "session_id": "session_1"}
                    
                    # Gọi API với stream=True để nhận NDJSON
                    res = requests.post(f"{API_URL}/chat", json=payload, stream=True)
                    
                    if res.status_code == 200:
                        st.session_state.current_docs = ""
                        
                        def generate_stream():
                            for line in res.iter_lines(decode_unicode=True):
                                if line:
                                    try:
                                        data = json.loads(line)
                                        if data.get("type") == "token":
                                            yield data.get("content", "")
                                        elif data.get("type") == "docs":
                                            st.session_state.current_docs = data.get("content", "")
                                    except Exception:
                                        pass
                        
                        # Thực hiện stream gõ chữ real-time ngay trong Container
                        answer = st.write_stream(generate_stream())
                        
                        # Cập nhật dữ liệu vào session_state sau khi stream hoàn tất
                        retrieved_docs = st.session_state.get("current_docs", "")
                        st.session_state.last_docs = retrieved_docs
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "docs": retrieved_docs
                        })
                        
                        # Đồng bộ hoá cột hiển thị tài liệu tham khảo bên phải
                        with docs_placeholder.container():
                            if retrieved_docs:
                                st.markdown(
                                    f'<div style="background-color: #f1f3f4; color: #202124; padding: 15px; border-radius: 8px; '
                                    f'border-left: 5px solid #0078D4; font-size: 0.95rem; line-height: 1.6; '
                                    f'max-height: 75vh; overflow-y: auto;">'
                                    f'{retrieved_docs.replace(chr(10), "<br>")}'
                                    f'</div>', 
                                    unsafe_allow_html=True
                                )
                            else:
                                st.warning("ℹ️ Không tìm thấy tài liệu cụ thể cho câu trả lời này.")
                    else:
                        try:
                            error_msg = res.json().get("detail", "Lỗi không xác định")
                        except:
                            error_msg = res.text
                        st.error(f"⚠️ Backend báo lỗi: {error_msg}")
                except Exception as e:
                    st.error(f"❌ Lỗi kết nối server API: {e}")

if __name__ == "__main__":
    page_setup()
    sidebar()
    chat_interface()
