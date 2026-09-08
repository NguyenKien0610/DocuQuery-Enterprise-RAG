from types import SimpleNamespace

import pytest

from frontend import app
from scripts import benchmark_docuquery


def test_api_headers_read_environment(monkeypatch):
    monkeypatch.setenv("DOCUQUERY_API_KEY", "secret")
    monkeypatch.setenv("DOCUQUERY_WORKSPACE_ID", "team-a")

    assert app._api_headers() == {
        "X-API-Key": "secret",
        "X-Workspace-ID": "team-a",
    }


def test_frontend_request_attaches_workspace_headers(monkeypatch):
    captured = {}
    monkeypatch.setenv("DOCUQUERY_API_KEY", "secret")
    monkeypatch.setenv("DOCUQUERY_WORKSPACE_ID", "team-a")
    monkeypatch.setattr(
        app.requests,
        "request",
        lambda method, url, **kwargs: captured.update(
            {"method": method, "url": url, "headers": kwargs["headers"]}
        ),
    )

    app._request("GET", "http://backend/documents")

    assert captured == {
        "method": "GET",
        "url": "http://backend/documents",
        "headers": {"X-API-Key": "secret", "X-Workspace-ID": "team-a"},
    }


def test_benchmark_request_attaches_workspace_headers(monkeypatch):
    captured = {}
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"status": "ok"},
    )
    monkeypatch.setattr(
        benchmark_docuquery.requests,
        "request",
        lambda method, url, **kwargs: captured.update(kwargs) or response,
    )

    payload = benchmark_docuquery._request_json(
        "GET",
        "http://backend/documents",
        api_key="secret",
        workspace_id="team-a",
    )

    assert payload == {"status": "ok"}
    assert captured["headers"] == {
        "X-API-Key": "secret",
        "X-Workspace-ID": "team-a",
    }


def test_display_file_name_removes_sha256_prefix():
    stored_name = f"{'a' * 64}_report.pdf"

    assert app._display_file_name(stored_name) == "report.pdf"


def test_display_file_name_removes_revision_prefix():
    assert app._display_file_name(f"{'b' * 32}_report.pdf") == "report.pdf"


def test_query_backend_passes_retrieval_only_without_dropping_auth_headers(monkeypatch):
    calls = []
    monkeypatch.setenv("DOCUQUERY_API_KEY", "secret")
    monkeypatch.setenv("DOCUQUERY_WORKSPACE_ID", "team-a")

    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {
            "answer": "Answer",
            "cached": True,
            "context": [],
            "status": "generated",
        },
    )

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return response

    monkeypatch.setattr(app.requests, "request", fake_request)

    result = app.query_backend("What is the answer?", retrieval_only=True)

    assert result == ("Answer", True, [], "generated")
    assert calls[0][2]["json"] == {
        "query": "What is the answer?",
        "retrieval_only": True,
    }
    assert calls[0][2]["headers"] == {
        "X-API-Key": "secret",
        "X-Workspace-ID": "team-a",
    }


def test_export_conversation_is_allowlisted_and_keeps_ordered_source_numbers():
    exported = app.export_conversation(
        [
            {"role": "user", "content": "Doanh thu là bao nhiêu?", "api_key": "secret"},
            {
                "role": "assistant",
                "content": "Evidence is listed below.",
                "status": "retrieved",
                "context": [
                    {
                        "source_file": r"C:\server\a.pdf",
                        "page_number": 2,
                        "text": "Revenue `2025`",
                        "similarity": 0.91,
                        "confidence": 0.99,
                    },
                    {
                        "source_file": "/srv/private/b.pdf",
                        "text": "Cost",
                        "similarity": 0.82,
                    },
                    {
                        "source_file": r"C:\server\a.pdf",
                        "text": "Margin",
                    },
                ],
                "server_path": "/srv/private/secret",
            },
        ]
    )

    assert "Doanh thu là bao nhiêu?" in exported
    assert "[Source 1]" in exported
    assert "[Source 2]" in exported
    assert "[Source 3]" in exported
    assert exported.index("[Source 1]") < exported.index("[Source 2]") < exported.index("[Source 3]")
    assert "a.pdf" in exported and "b.pdf" in exported
    assert "C:\\server" not in exported
    assert "/srv/private" not in exported
    assert "secret" not in exported
    assert "confidence:" not in exported.lower()
    assert "0.99" not in exported
    assert "similarity: 0.91" in exported
    assert "```text" in exported


def test_export_conversation_empty_session_is_empty_markdown():
    assert app.export_conversation([]) == ""


@pytest.mark.parametrize("status,label", [
    ("retrieved", "no generated answer"), ("degraded", "generation unavailable"),
    ("insufficient_context", "insufficient evidence"), ("generated", "generated answer"),
])
def test_export_labels_each_answer_status_and_fences_untrusted_text(status, label):
    exported = app.export_conversation([
        {"role": "assistant", "status": status, "content": "```\n<script>not markup</script>\n```",
         "context": [{"source_file": "[click](https://example.invalid).txt", "page_number": 3,
                      "text": "```\n# fake heading\n```", "private_path": "/private/secret"}]},
    ])
    assert label in exported.lower()
    assert "````text" in exported
    assert "page: 3" in exported
    assert "/private/secret" not in exported
