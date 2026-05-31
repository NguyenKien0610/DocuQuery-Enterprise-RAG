import hashlib
import re
import time

import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/api/v1/documents/upload"
DOCUMENTS_ENDPOINT = f"{API_BASE_URL}/api/v1/documents"
STATUS_ENDPOINT_TEMPLATE = f"{API_BASE_URL}/api/v1/documents/status" + "/{task_id}"
QUERY_ENDPOINT = f"{API_BASE_URL}/api/v1/query"
RESET_ENDPOINT = f"{API_BASE_URL}/api/v1/workspace/reset"
REQUEST_TIMEOUT = 120
STREAM_DELAY_SECONDS = 0.03
TASK_POLL_INTERVAL_SECONDS = 2
TASK_POLL_TIMEOUT_SECONDS = 300


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sidebar_notice" not in st.session_state:
        st.session_state.sidebar_notice = None
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None
    if "last_uploaded_signature" not in st.session_state:
        st.session_state.last_uploaded_signature = None
    if "pending_upload" not in st.session_state:
        st.session_state.pending_upload = None


def _uploaded_file_signature(uploaded_file) -> str:
    file_bytes = uploaded_file.getvalue()
    digest = hashlib.sha256(file_bytes).hexdigest()
    return f"{uploaded_file.name}:{len(file_bytes)}:{digest}"


def start_upload(uploaded_file) -> str | None:
    file_content_type = uploaded_file.type or "application/octet-stream"
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), file_content_type)}
    try:
        response = requests.post(
            UPLOAD_ENDPOINT,
            files=files,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        task_id = payload.get("task_id")
        if not task_id:
            st.sidebar.error("Backend returned no task_id for the uploaded document.")
            return None

        return str(task_id)
    except requests.exceptions.RequestException as exc:
        st.sidebar.error(f"Cannot connect to backend API: {exc}")
    except ValueError:
        st.sidebar.error("Backend returned an invalid JSON response.")
    return None


def poll_upload_task(task_id: str) -> bool:
    started_at = time.monotonic()

    progress_bar = st.sidebar.progress(
        5,
        text="Đang phân tích tài liệu (Celery Worker)...",
    )
    progress_value = 5

    try:
        with st.spinner("Đang phân tích tài liệu (Celery Worker)..."):
            while True:
                if time.monotonic() - started_at > TASK_POLL_TIMEOUT_SECONDS:
                    progress_bar.empty()
                    st.session_state.sidebar_notice = (
                        "warning",
                        "Document processing is taking longer than expected. Please wait and try again shortly.",
                    )
                    return False

                status_response = requests.get(
                    STATUS_ENDPOINT_TEMPLATE.format(task_id=task_id),
                    timeout=REQUEST_TIMEOUT,
                )
                status_response.raise_for_status()
                status_payload = status_response.json()
                status = str(status_payload.get("status", "")).upper()

                if status == "SUCCESS":
                    progress_bar.progress(100, text="Phân tích tài liệu hoàn tất.")
                    st.session_state.sidebar_notice = (
                        "success",
                        f"Document processed successfully. Task ID: {task_id}",
                    )
                    return True

                if status == "FAILURE":
                    progress_bar.empty()
                    error_message = status_payload.get("error") or "Celery worker failed to process the document."
                    st.sidebar.error(f"Document processing failed: {error_message}")
                    return False

                progress_value = min(progress_value + 10, 95)
                progress_bar.progress(
                    progress_value,
                    text=f"Đang phân tích tài liệu (Celery Worker)... [{status or 'PENDING'}]",
                )
                time.sleep(TASK_POLL_INTERVAL_SECONDS)
    except requests.exceptions.RequestException as exc:
        progress_bar.empty()
        st.sidebar.error(f"Cannot connect to backend API: {exc}")
    except ValueError:
        progress_bar.empty()
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


def query_backend(question: str) -> tuple[str, bool, list[dict[str, object]]]:
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
        [dict(chunk) for chunk in payload.get("context", []) if isinstance(chunk, dict)],
    )


def reset_backend_workspace() -> None:
    response = requests.delete(
        RESET_ENDPOINT,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()


def stream_answer(answer: str):
    for chunk in re.split(r"(\s+)", answer):
        if not chunk:
            continue
        yield chunk
        if not chunk.isspace():
            time.sleep(STREAM_DELAY_SECONDS)


def _display_file_name(file_name: str) -> str:
    return re.sub(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}_",
        "",
        file_name,
    )


def _format_context_chunk(chunk: str) -> str:
    cleaned = chunk.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"(?m)^\s*\d+\s+", "", cleaned)
    cleaned = re.sub(r"(?<!\S)\d+(?!\S)", " ", cleaned)
    cleaned = re.sub(r"([^\.\!\?])\n", r"\1 ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    normalized_lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return "\n".join(normalized_lines).strip()


def render_context_chunks(context: list[dict[str, object]]) -> None:
    with st.expander("🔍 Xem trích dẫn nguồn (Context)"):
        grouped_chunks: dict[str, list[dict[str, object]]] = {}

        for idx, chunk in enumerate(context, start=1):
            if not isinstance(chunk, dict):
                chunk = {
                    "source_file": "Unknown source",
                    "chunk_index": idx - 1,
                    "text": str(chunk),
                }
            source_file = str(chunk.get("source_file", "Unknown source"))
            grouped_chunks.setdefault(source_file, []).append(chunk)

        total_files = len(grouped_chunks)
        for file_idx, (source_file, chunks) in enumerate(grouped_chunks.items(), start=1):
            display_name = _display_file_name(source_file)
            st.markdown(f"### 📁 {display_name}")

            for chunk_idx, chunk in enumerate(chunks, start=1):
                raw_chunk_index = chunk.get("chunk_index", chunk_idx - 1)
                try:
                    chunk_number = int(raw_chunk_index) + 1
                except (TypeError, ValueError):
                    chunk_number = chunk_idx

                formatted_chunk = _format_context_chunk(str(chunk.get("text", "")))
                st.markdown(f"**📄 Trích đoạn {chunk_number}:**")
                page_number = chunk.get("page_number")
                if page_number is not None:
                    st.caption(f"Trang: {page_number}")
                if formatted_chunk:
                    blockquote = "\n".join(f"> {line}" for line in formatted_chunk.splitlines())
                    st.markdown(blockquote)
                else:
                    st.info("Không có nội dung khả dụng.")

                if chunk_idx < len(chunks):
                    st.divider()

            if file_idx < total_files:
                st.markdown("---")


def render_sidebar() -> None:
    st.sidebar.header("Document Upload")
    if st.session_state.sidebar_notice:
        level, message = st.session_state.sidebar_notice
        if level == "success":
            st.sidebar.success(message)
        elif level == "warning":
            st.sidebar.warning(message)
        else:
            st.sidebar.error(message)
        st.session_state.sidebar_notice = None

    uploaded_file = st.sidebar.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "txt"],
    )

    if uploaded_file is not None:
        current_signature = _uploaded_file_signature(uploaded_file)
        pending_upload = st.session_state.pending_upload

        if (
            pending_upload
            and pending_upload.get("signature") == current_signature
            and pending_upload.get("task_id")
        ):
            if poll_upload_task(str(pending_upload["task_id"])):
                st.session_state.last_uploaded_file = uploaded_file.name
                st.session_state.last_uploaded_signature = current_signature
                st.session_state.pending_upload = None
                st.rerun()
        elif current_signature != st.session_state.last_uploaded_signature:
            task_id = start_upload(uploaded_file)
            if task_id:
                st.session_state.pending_upload = {
                    "task_id": task_id,
                    "signature": current_signature,
                    "file_name": uploaded_file.name,
                }
                if poll_upload_task(task_id):
                    st.session_state.last_uploaded_file = uploaded_file.name
                    st.session_state.last_uploaded_signature = current_signature
                    st.session_state.pending_upload = None
                    st.rerun()

    st.sidebar.subheader("📂 Tài liệu đang trong hệ thống")
    try:
        documents = fetch_documents()
        if documents:
            for document in documents:
                st.sidebar.markdown(f"- {_display_file_name(document)}")
        else:
            st.sidebar.caption("Chưa có tài liệu nào")
    except requests.exceptions.RequestException as exc:
        st.sidebar.error(f"Cannot connect to backend API: {exc}")
    except ValueError:
        st.sidebar.error("Backend returned an invalid JSON response.")

    if st.sidebar.button("🗑️ Tạo phiên Chat mới (Xóa dữ liệu)", use_container_width=True):
        try:
            reset_backend_workspace()
            st.session_state.messages = []
            st.session_state.last_uploaded_file = None
            st.session_state.last_uploaded_signature = None
            st.session_state.pending_upload = None
            st.session_state.sidebar_notice = (
                "success",
                "Workspace reset completed. All uploaded documents were removed.",
            )
            st.rerun()
        except requests.exceptions.RequestException as exc:
            st.sidebar.error(f"Cannot connect to backend API: {exc}")


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("cached"):
                st.caption("⚡ Cached")
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("context"):
                render_context_chunks(message["context"])


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
                render_context_chunks(context)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": streamed_answer,
                    "cached": cached,
                    "context": context,
                }
            )
        except requests.exceptions.RequestException as exc:
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
