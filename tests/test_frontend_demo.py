from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_streamlit_demo_mode_status_unicode_and_download(monkeypatch):
    calls = []
    downloads = []
    original_download = st.download_button

    def download(*args, **kwargs):
        downloads.append(args[1])
        return original_download(*args, **kwargs)

    monkeypatch.setattr(st, "download_button", download)

    def fake_request(method, url, **kwargs):
        if method == "GET":
            return _Response({"documents": ["demo.pdf"]})
        calls.append((method, url, kwargs))
        retrieval_only = kwargs["json"]["retrieval_only"]
        if retrieval_only:
            return _Response(
                {
                    "answer": "",
                    "cached": False,
                    "status": "retrieved",
                    "context": [
                        {
                            "source_file": r"C:\server\demo.pdf",
                            "page_number": 4,
                            "text": "Doanh thu năm 2025",
                        }
                    ],
                }
            )
        return _Response(
            {
                "answer": "Đáp án từ tài liệu [Source 1]",
                "cached": False,
                "status": "generated",
                "context": [],
            }
        )

    monkeypatch.setattr("requests.request", fake_request)
    app_path = Path(__file__).parents[1] / "frontend" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=10).run()

    assert len(at.radio) == 1
    at.radio[0].set_value("Evidence only").run()
    at.chat_input[0].set_value("Doanh thu là bao nhiêu?").run()

    assert calls[-1][2]["json"] == {
        "query": "Doanh thu là bao nhiêu?",
        "retrieval_only": True,
    }
    assert any("no generated answer" in item.value.lower() for item in at.info)
    assert any("Doanh thu năm 2025" in item.value for item in at.code)
    assert not at.exception
    assert len(at.get("download_button")) == 1
    assert "Doanh thu là bao nhiêu?" in downloads[-1]
    assert "[Source 1]" in downloads[-1]
    assert "page: 4" in downloads[-1]
    assert "C:\\server" not in downloads[-1]

    at.radio[0].set_value("Answer generation").run()
    assert any("no generated answer" in item.value.lower() for item in at.info)
    assert len(at.get("download_button")) == 1
    at.chat_input[0].set_value("What is the answer?").run()

    assert calls[-1][2]["json"] == {
        "query": "What is the answer?",
        "retrieval_only": False,
    }
    assert any("Đáp án từ tài liệu" in item.value for item in at.markdown)
    assert not at.exception


@pytest.mark.parametrize("status,element,label", [
    ("degraded", "warning", "Generation unavailable"),
    ("insufficient_context", "info", "Insufficient evidence"),
    ("retrieved", "info", "no generated answer"),
])
def test_status_and_interleaved_citations_survive_rerun(monkeypatch, status, element, label):
    calls = []
    context = [
        {"source_file": "/srv/a.pdf", "text": "A first", "page_number": 2},
        {"source_file": r"C:\private\b.pdf", "text": "B only", "page_number": 5},
        {"source_file": "/srv/a.pdf", "text": "<script>literal A second</script>", "page_number": 3},
    ]

    def request(method, url, **kwargs):
        if method == "GET":
            return _Response({"documents": []})
        calls.append(kwargs["json"])
        return _Response({"status": status, "answer": "", "cached": False, "context": context})

    monkeypatch.setattr("requests.request", request)
    monkeypatch.setattr(st, "write_stream", lambda *args: pytest.fail("Non-generated response was streamed"))
    app_path = Path(__file__).parents[1] / "frontend" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=10).run()
    at.chat_input[0].set_value("Kiểm tra nguồn").run()
    for _ in range(2):
        assert not at.exception
        assert any(label in item.value for item in at.get(element))
        citations = [item.value for item in at.markdown if "[Source " in item.value]
        assert len(citations) == 3
        assert "[Source 1]" in citations[0]
        assert "[Source 3]" in citations[1]
        assert "[Source 2]" in citations[2]
        assert any("literal A second" in item.value for item in at.code)
        assert any("page: 3" in item.value for item in at.caption)
        assert not any("/srv/" in item.value or "C:\\private" in item.value for item in at.text)
        assert len(at.get("download_button")) == 1
        at.run()
    assert len(calls) == 1


def test_delete_control_requires_confirmation_and_sends_exact_revision(monkeypatch):
    documents = [{"document_id": "a" * 64, "revision": "1" * 32, "source_file": "notes.txt", "bytes": 5, "chunks": 1}]
    deletes = []
    monkeypatch.setenv("DOCUQUERY_API_KEY", "owner-test")
    monkeypatch.setenv("DOCUQUERY_WORKSPACE_ID", "team-a")

    def request(method, url, **kwargs):
        if method == "GET" and url.endswith("/managed"):
            return _Response({"items": list(documents)})
        if method == "GET":
            return _Response({"documents": [doc["source_file"] for doc in documents]})
        assert method == "DELETE" and not url.endswith("/reset")
        deletes.append((url, kwargs))
        documents.clear()
        return _Response({"deleted": True, "cleanup_pending": True})

    monkeypatch.setattr("requests.request", request)
    at = AppTest.from_file(str(Path(__file__).parents[1] / "frontend" / "app.py"), default_timeout=10).run()
    button = at.button(key="delete_one_document")
    assert button.disabled
    assert deletes == []
    at.checkbox[0].check().run()
    assert not at.button(key="delete_one_document").disabled
    at.button(key="delete_one_document").click().run()
    assert not at.exception
    assert len(deletes) == 1
    assert deletes[0][0].endswith("/" + "a" * 64)
    assert deletes[0][1]["params"] == {"revision": "1" * 32}
    assert deletes[0][1]["headers"] == {"X-API-Key": "owner-test", "X-Workspace-ID": "team-a"}
    assert any("cleanup" in item.value.lower() for item in at.warning)
    assert len(at.selectbox) == 0
