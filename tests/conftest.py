import pytest


@pytest.fixture(autouse=True)
def isolated_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    monkeypatch.setenv("DOCUQUERY_STATE_DIR", str(tmp_path / ".state"))
    monkeypatch.setenv("DOCUQUERY_WORKSPACE_ID", "team-a")
    monkeypatch.delenv("DOCUQUERY_CREDENTIALS", raising=False)
