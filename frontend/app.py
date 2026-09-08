import hashlib
import math
import os
import re
import time

import requests
import streamlit as st

API_BASE_URL = os.getenv("DOCUQUERY_API_BASE_URL", "http://localhost:8000").rstrip("/")
UPLOAD_ENDPOINT = f"{API_BASE_URL}/api/v1/documents/upload"
DOCUMENTS_ENDPOINT = f"{API_BASE_URL}/api/v1/documents"
STATUS_ENDPOINT_TEMPLATE = f"{API_BASE_URL}/api/v1/documents/status" + "/{task_id}"
QUERY_ENDPOINT = f"{API_BASE_URL}/api/v1/query"
RESET_ENDPOINT = f"{API_BASE_URL}/api/v1/workspace/reset"
REQUEST_TIMEOUT = 120
STREAM_DELAY_SECONDS = 0.03
TASK_POLL_INTERVAL_SECONDS = 2
TASK_POLL_TIMEOUT_SECONDS = 300
STATUS_LABELS = {
    "generated": "Generated answer",
    "retrieved": "Evidence only — no generated answer",
    "degraded": "Generation unavailable — not a complete generated answer",
    "insufficient_context": "Insufficient evidence — no supported answer",
    "error": "Request failed",
}


def _api_headers() -> dict[str, str]:
    return {
        "X-API-Key": os.getenv("DOCUQUERY_API_KEY", ""),
        "X-Workspace-ID": os.getenv("DOCUQUERY_WORKSPACE_ID", "default"),
    }


def _request(method: str, url: str, **kwargs):
    supplied_headers = kwargs.pop("headers", {})
    headers = {**supplied_headers, **_api_headers()}
    return requests.request(method, url, headers=headers, **kwargs)


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
        response = _request(
            "POST",
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

                status_response = _request(
                    "GET",
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
    response = _request(
        "GET",
        DOCUMENTS_ENDPOINT,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    documents = payload.get("documents", [])
    return [str(document) for document in documents]


def query_backend(question: str, *, retrieval_only: bool = False) -> tuple[str, bool, list[dict[str, object]], str]:
    response = _request(
        "POST",
        QUERY_ENDPOINT,
        json={"query": question, "retrieval_only": retrieval_only},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    return (
        payload.get("answer", ""),
        bool(payload.get("cached", False)),
        [dict(chunk) for chunk in payload.get("context", []) if isinstance(chunk, dict)],
        str(payload.get("status", "generated")),
    )


def reset_backend_workspace() -> None:
    response = _request(
        "DELETE",
        RESET_ENDPOINT,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()


def render_document_delete_controls() -> None:
    try:
        response = _request("GET", f"{DOCUMENTS_ENDPOINT}/managed", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        documents = response.json().get("items", [])
    except (requests.exceptions.RequestException, ValueError):
        st.sidebar.caption("Document management is temporarily unavailable.")
        return
    if not documents:
        return
    with st.sidebar.expander("Delete one document (owner only)"):
        by_revision = {doc["revision"]: doc for doc in documents}
        selected = st.selectbox(
            "Document to delete", list(by_revision),
            format_func=lambda revision: _display_file_name(str(by_revision[revision]["source_file"])),
            key="delete_document_revision",
        )
        if selected is None:
            return
        confirmed = st.checkbox(
            "I confirm deletion of this document and its indexed content.",
            key=f"confirm_delete_{selected}",
        )
        st.caption("Previous chat and exports are historical copies. Active uploads must finish first.")
        if st.button("Delete selected document", key="delete_one_document", disabled=not confirmed) and confirmed:
            document = by_revision[selected]
            try:
                response = _request(
                    "DELETE", f"{DOCUMENTS_ENDPOINT}/{document['document_id']}",
                    params={"revision": selected}, timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                cleanup_pending = bool(response.json().get("cleanup_pending"))
                st.session_state.sidebar_notice = (
                    "warning" if cleanup_pending else "success",
                    "Document removed from search. Physical cleanup is pending; an owner can reconcile."
                    if cleanup_pending else "Document removed from search. Previous chat remains historical.",
                )
                st.rerun()
            except requests.exceptions.RequestException as exc:
                status_code = exc.response.status_code if exc.response is not None else 0
                st.error({
                    403: "Only a workspace owner can delete documents.",
                    404: "Document no longer exists. Refresh the document list.",
                    409: "Document changed or workspace has active uploads. Refresh and try again after processing.",
                }.get(status_code, "Document deletion is temporarily unavailable. Please retry."))
            except ValueError:
                st.error("Backend returned an invalid deletion response. Refresh the document list.")


def stream_answer(answer: str):
    for chunk in re.split(r"(\s+)", answer):
        if not chunk:
            continue
        yield chunk
        if not chunk.isspace():
            time.sleep(STREAM_DELAY_SECONDS)


def _display_file_name(file_name: str) -> str:
    file_name = file_name.replace("\\", "/").rsplit("/", 1)[-1]
    return re.sub(
        r"^(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{64}|[0-9a-fA-F]{8}-[0-9a-fA-F-]{27})_",
        "",
        file_name,
    )


def _format_context_chunk(chunk: str) -> str:
    return chunk.replace("\r\n", "\n").replace("\r", "\n").strip()


def _text_block(text: str) -> str:
    """Keep arbitrary document/chat Markdown inside a literal fenced block."""
    fence = "`" * max(3, max((len(match) + 1 for match in re.findall(r"`+", text)), default=0))
    return f"{fence}text\n{text}\n{fence}"


def _citation_details(chunk: dict[str, object]) -> str:
    details = []
    page = chunk.get("page_number")
    if type(page) is int and page > 0:
        details.append(f"page: {page}")
    similarity = chunk.get("similarity")
    if isinstance(similarity, (int, float)) and not isinstance(similarity, bool) and math.isfinite(similarity):
        details.append(f"similarity: {similarity:.2f} (not confidence)")
    return "; ".join(details)


def export_conversation(messages: list[dict]) -> str:
    """Export current-session content and an explicit allowlist of citation fields."""
    sections = []
    for message in messages:
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        sections.append(f"## {role.title()}")
        if role == "assistant":
            status = str(message.get("status", "unknown"))
            sections.append(f"Status: {STATUS_LABELS.get(status, 'Unknown response status')}")
        content = message.get("content", "")
        if isinstance(content, str) and content:
            sections.append(_text_block(content))
        if role != "assistant":
            continue
        for index, chunk in enumerate(message.get("context", []), 1):
            if not isinstance(chunk, dict):
                continue
            filename = _display_file_name(str(chunk.get("source_file", "Unknown source")))
            sections.append(f"### [Source {index}]")
            sections.append(_text_block(filename))
            details = _citation_details(chunk)
            if details:
                sections.append(details)
            text = chunk.get("text", "")
            if isinstance(text, str) and text:
                sections.append(_text_block(text))
    return "\n\n".join(sections) + ("\n" if sections else "")


def render_context_chunks(context: list[dict[str, object]]) -> None:
    with st.expander("🔍 Xem trích dẫn nguồn (Context)"):
        grouped_chunks: dict[str, list[tuple[int, dict[str, object]]]] = {}

        for idx, chunk in enumerate(context, start=1):
            if not isinstance(chunk, dict):
                chunk = {
                    "source_file": "Unknown source",
                    "chunk_index": idx - 1,
                    "text": str(chunk),
                }
            source_file = str(chunk.get("source_file", "Unknown source"))
            grouped_chunks.setdefault(source_file, []).append((idx, chunk))

        total_files = len(grouped_chunks)
        for file_idx, (source_file, chunks) in enumerate(grouped_chunks.items(), start=1):
            display_name = _display_file_name(source_file)
            st.text(f"📁 {display_name}")

            for chunk_idx, (source_number, chunk) in enumerate(chunks, start=1):
                raw_chunk_index = chunk.get("chunk_index", chunk_idx - 1)
                try:
                    chunk_number = (
                        int(raw_chunk_index) + 1
                        if isinstance(raw_chunk_index, (int, str))
                        else chunk_idx
                    )
                except ValueError:
                    chunk_number = chunk_idx

                formatted_chunk = _format_context_chunk(str(chunk.get("text", "")))
                st.markdown(f"**[Source {source_number}] — Trích đoạn {chunk_number}**")
                details = _citation_details(chunk)
                if details:
                    st.caption(details)
                if formatted_chunk:
                    st.code(formatted_chunk, language="text", wrap_lines=True)
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

    render_document_delete_controls()

    if st.sidebar.button("🗑️ Tạo phiên Chat mới (Xóa dữ liệu)", use_container_width=True):
        try:
            reset_backend_workspace()
            st.session_state.messages = []
            st.session_state.last_uploaded_file = None
            st.session_state.last_uploaded_signature = None
            st.session_state.pending_upload = None
            st.session_state.sidebar_notice = (
                "success",
                "Workspace reset completed. Documents are no longer searchable; "
                "physical cleanup may still be pending.",
            )
            st.rerun()
        except requests.exceptions.RequestException as exc:
            st.sidebar.error(f"Cannot connect to backend API: {exc}")


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(message)
            else:
                st.markdown(message["content"])


def render_assistant_message(message: dict, *, stream: bool = False) -> None:
    status = message.get("status", "generated")
    if status in ("degraded", "error"):
        st.warning(STATUS_LABELS[status])
    elif status in ("retrieved", "insufficient_context"):
        st.info(STATUS_LABELS[status])
    if message.get("cached"):
        st.caption("⚡ Cached")
    content = message.get("content", "")
    if content and status != "retrieved":
        if stream:
            st.write_stream(stream_answer(content))
        else:
            st.markdown(content)
    if message.get("context"):
        render_context_chunks(message["context"])


def handle_question(question: str, *, retrieval_only: bool) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            answer, cached, context, status = query_backend(question, retrieval_only=retrieval_only)
            message = {
                "role": "assistant", "content": answer if status != "retrieved" else "",
                "cached": cached, "context": context, "status": status,
            }
            render_assistant_message(message, stream=status == "generated")
            st.session_state.messages.append(message)
        except requests.exceptions.RequestException:
            error_message = "Cannot connect to backend API. Please try again."
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message, "context": [], "status": "error"}
            )
        except ValueError:
            error_message = "Backend returned an invalid JSON response."
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message, "context": [], "status": "error"}
            )


def main() -> None:
    st.set_page_config(page_title="DocuQuery Frontend", page_icon="📄", layout="wide")
    init_session_state()
    st.title("DocuQuery")
    st.caption("Upload PDF, DOCX or TXT documents, then inspect evidence or generate an answer.")
    mode = st.radio("Query mode", ["Answer generation", "Evidence only"], horizontal=True)
    if mode == "Evidence only":
        st.caption("Search evidence without Gemini or the answer cache. Similarity is not answer confidence.")
    render_sidebar()
    render_chat_history()
    question = st.chat_input("Ask a question about your uploaded documents")
    if question:
        handle_question(question, retrieval_only=mode == "Evidence only")
    if st.session_state.messages:
        st.download_button(
            "Download conversation (.md)", export_conversation(st.session_state.messages),
            file_name="docuquery-conversation.md", mime="text/markdown; charset=utf-8",
            key="download_conversation", on_click="ignore",
        )


if __name__ == "__main__":
    main()
