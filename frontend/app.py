import time

import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/api/v1/documents/upload"
DOCUMENTS_ENDPOINT = f"{API_BASE_URL}/api/v1/documents"
QUERY_ENDPOINT = f"{API_BASE_URL}/api/v1/query"
RESET_ENDPOINT = f"{API_BASE_URL}/api/v1/workspace/reset"
REQUEST_TIMEOUT = 120
STREAM_DELAY_SECONDS = 0.03


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sidebar_notice" not in st.session_state:
        st.session_state.sidebar_notice = None


def upload_document(uploaded_file) -> bool:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
    try:
        response = requests.post(
            UPLOAD_ENDPOINT,
            files=files,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        st.session_state.sidebar_notice = (
            "success",
            f"Upload queued. Task ID: {payload.get('task_id', 'unknown')}",
        )
        return True
    except requests.RequestException as exc:
        st.sidebar.error(f"Cannot connect to backend API: {exc}")
    except ValueError:
        st.sidebar.error("Backend returned an invalid JSON response.")
    return False


def fetch_documents() -> list[str]:
    response = requests.get(
        DOCUMENTS_ENDPOINT,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    documents = payload.get("documents", [])
    return [str(document) for document in documents]


def query_backend(question: str) -> tuple[str, bool, list[str]]:
    response = requests.post(
        QUERY_ENDPOINT,
        json={"query": question},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    return (
        payload.get("answer", ""),
        bool(payload.get("cached", False)),
        [str(chunk) for chunk in payload.get("context", [])],
    )


def reset_backend_workspace() -> None:
    response = requests.delete(
        RESET_ENDPOINT,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()


def stream_answer(answer: str):
    for word in answer.split():
        yield f"{word} "
        time.sleep(STREAM_DELAY_SECONDS)


def render_sidebar() -> None:
    st.sidebar.header("Document Upload")
    if st.session_state.sidebar_notice:
        level, message = st.session_state.sidebar_notice
        if level == "success":
            st.sidebar.success(message)
        else:
            st.sidebar.error(message)
        st.session_state.sidebar_notice = None

    uploaded_file = st.sidebar.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "txt"],
    )

    if uploaded_file is not None:
        if st.sidebar.button("Send to Backend", use_container_width=True):
            if upload_document(uploaded_file):
                st.rerun()

    st.sidebar.subheader("📂 Tài liệu đang trong hệ thống")
    try:
        documents = fetch_documents()
        if documents:
            for document in documents:
                st.sidebar.markdown(f"- {document}")
        else:
            st.sidebar.caption("Chưa có tài liệu nào")
    except requests.RequestException as exc:
        st.sidebar.error(f"Cannot connect to backend API: {exc}")
    except ValueError:
        st.sidebar.error("Backend returned an invalid JSON response.")

    if st.sidebar.button("🗑️ Tạo phiên Chat mới (Xóa dữ liệu)", use_container_width=True):
        try:
            reset_backend_workspace()
            st.session_state.messages = []
            st.session_state.sidebar_notice = (
                "success",
                "Workspace reset completed. All uploaded documents were removed.",
            )
            st.rerun()
        except requests.RequestException as exc:
            st.sidebar.error(f"Cannot connect to backend API: {exc}")


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("cached"):
                st.caption("⚡ Cached")
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("context"):
                with st.expander("🔍 Xem trích dẫn nguồn (Context)"):
                    for idx, chunk in enumerate(message["context"], start=1):
                        st.markdown(f"**Chunk {idx}**")
                        st.markdown(chunk)


def main() -> None:
    st.set_page_config(page_title="DocuQuery Frontend", page_icon="📄", layout="wide")
    init_session_state()

    st.title("DocuQuery")
    st.caption("Upload PDF documents and query them through the FastAPI backend.")

    render_sidebar()
    render_chat_history()

    question = st.chat_input("Ask a question about your uploaded documents")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            answer, cached, context = query_backend(question)
            if cached:
                st.caption("⚡ Cached")
            streamed_answer = st.write_stream(stream_answer(answer))
            if context:
                with st.expander("🔍 Xem trích dẫn nguồn (Context)"):
                    for idx, chunk in enumerate(context, start=1):
                        st.markdown(f"**Chunk {idx}**")
                        st.markdown(chunk)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": streamed_answer,
                    "cached": cached,
                    "context": context,
                }
            )
        except requests.RequestException as exc:
            error_message = f"Cannot connect to backend API: {exc}"
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message, "context": []}
            )
        except ValueError:
            error_message = "Backend returned an invalid JSON response."
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message, "context": []}
            )


if __name__ == "__main__":
    main()
