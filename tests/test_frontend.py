from types import SimpleNamespace

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
