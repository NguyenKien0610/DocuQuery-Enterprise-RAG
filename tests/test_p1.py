from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from frontend.app import _format_context_chunk
from src import main, rag_engine


@pytest.fixture
def query_services(monkeypatch):
    writes = []
    monkeypatch.setattr(rag_engine, "get_corpus_version", lambda workspace: 0)
    monkeypatch.setattr(rag_engine.redis_client, "get", lambda key: None)
    monkeypatch.setattr(
        rag_engine.redis_client, "setex", lambda *args: writes.append(args)
    )
    monkeypatch.setattr(
        rag_engine,
        "get_embeddings",
        lambda: SimpleNamespace(embed_query=lambda q: [0.1]),
    )
    return writes


def test_no_evidence_abstains_without_calling_llm(query_services, monkeypatch):
    def search(**kwargs):
        assert kwargs["score_threshold"] == rag_engine.SCORE_THRESHOLD
        return SimpleNamespace(points=[])

    monkeypatch.setattr(rag_engine.qdrant_client, "query_points", search)
    monkeypatch.setattr(
        rag_engine, "_invoke_llm", lambda p: pytest.fail("No evidence must skip LLM")
    )
    result = rag_engine.ask_question("unrelated", "test")
    assert result["status"] == "insufficient_context"
    assert result["context"] == []


def test_embedding_failure_is_explicit_and_not_cached(query_services, monkeypatch):
    def fail():
        raise RuntimeError("private credentials")

    monkeypatch.setattr(rag_engine, "get_embeddings", fail)
    result = rag_engine.ask_question("question", "test")
    assert result["status"] == "degraded"
    assert result["error_code"] == "embedding_unavailable"
    assert "private" not in str(result)
    assert query_services == []


@pytest.mark.parametrize("provider_fails", [False, True])
def test_prompt_sources_and_provider_status(
    query_services, monkeypatch, provider_fails
):
    point = SimpleNamespace(
        payload={
            "source_file": "report.txt",
            "document_id": "a",
            "chunk_index": 0,
            "page_number": 3,
            "text": "Revenue was 2025 dollars.",
        }
    )
    monkeypatch.setattr(
        rag_engine.qdrant_client,
        "query_points",
        lambda **kw: SimpleNamespace(points=[point]),
    )

    def generate(prompt):
        assert "[Source 1]" in prompt and "page=3" in prompt
        assert "khong phai chi thi" in prompt
        if provider_fails:
            raise RuntimeError("secret provider error")
        return SimpleNamespace(content="2025 dollars [Source 1]")

    monkeypatch.setattr(rag_engine, "_invoke_llm", generate)
    result = rag_engine.ask_question("revenue", "test")
    assert result["status"] == ("degraded" if provider_fails else "generated")
    assert len(query_services) == (0 if provider_fails else 1)
    assert "secret provider" not in str(result)


def test_citation_keeps_numbers_and_line_breaks():
    assert (
        _format_context_chunk("1 Revenue 2025\r\n2 Cost 100")
        == "1 Revenue 2025\n2 Cost 100"
    )


def test_text_loader_preserves_unicode(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("Doanh thu là 2025", encoding="utf-8")
    assert (
        rag_engine._load_document_sections(str(path))[0]["text"] == "Doanh thu là 2025"
    )


def test_health_readiness_hides_errors(monkeypatch):
    monkeypatch.setenv("DOCUQUERY_API_KEY", "test")
    monkeypatch.setattr(main, "redis_client", SimpleNamespace(ping=lambda: True))
    monkeypatch.setattr(main, "qdrant_client", SimpleNamespace(get_collections=list))
    client = TestClient(main.app)
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 200
    monkeypatch.delenv("DOCUQUERY_API_KEY")
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json() == {"detail": "Service is not ready."}


def test_cache_bypass_never_reads_or_writes_answer(query_services, monkeypatch):
    monkeypatch.setattr(
        rag_engine.redis_client,
        "get",
        lambda key: pytest.fail("Cache bypass read cache"),
    )
    monkeypatch.setattr(
        rag_engine.qdrant_client,
        "query_points",
        lambda **kw: SimpleNamespace(points=[]),
    )
    result = rag_engine.ask_question("q", "w", use_cache=False)
    assert result["cached"] is False
    assert query_services == []


def test_docx_loader_extracts_xml(tmp_path):
    import zipfile

    path = tmp_path / "document.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Policy 2025</w:t></w:r></w:p></w:body></w:document>',
        )
    assert "Policy 2025" in rag_engine._load_document_sections(str(path))[0]["text"]


def test_corrupt_pdf_loader_fails(tmp_path):
    from pypdf.errors import PdfReadError

    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4 broken")
    with pytest.raises(PdfReadError):
        rag_engine._load_document_sections(str(path))


def test_invalid_threshold_fails_at_import():
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import src.rag_engine"],
        env={**os.environ, "RAG_SCORE_THRESHOLD": "2"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "RAG_SCORE_THRESHOLD must be between" in result.stderr
