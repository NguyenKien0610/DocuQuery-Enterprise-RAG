import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/api/v1/documents/upload"
QUERY_ENDPOINT = f"{API_BASE_URL}/api/v1/query"
REQUEST_TIMEOUT = 120


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def upload_document(uploaded_file) -> None:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
    try:
        response = requests.post(
            UPLOAD_ENDPOINT,
            files=files,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        st.sidebar.success(f"Upload queued. Task ID: {payload.get('task_id', 'unknown')}")
    except requests.RequestException as exc:
        st.sidebar.error(f"Cannot connect to backend API: {exc}")
    except ValueError:
        st.sidebar.error("Backend returned an invalid JSON response.")


def query_backend(question: str) -> tuple[str, bool]:
    response = requests.post(
        QUERY_ENDPOINT,
        json={"query": question},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("answer", ""), bool(payload.get("cached", False))


def render_sidebar() -> None:
    st.sidebar.header("Document Upload")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.sidebar.button("Send to Backend", use_container_width=True):
            upload_document(uploaded_file)


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


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
            answer, cached = query_backend(question)
            if cached:
                st.caption("⚡ Cached")
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except requests.RequestException as exc:
            error_message = f"Cannot connect to backend API: {exc}"
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message}
            )
        except ValueError:
            error_message = "Backend returned an invalid JSON response."
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message}
            )


if __name__ == "__main__":
    main()
