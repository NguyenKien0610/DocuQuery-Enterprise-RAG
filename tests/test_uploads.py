import io
import zipfile

import pytest
from fastapi import UploadFile

from src.uploads import UploadRejected, save_validated_upload


def make_docx(text: str = "hello") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr("word/document.xml", f"<document>{text}</document>")
    return buffer.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("empty.pdf", b""),
        ("fake.pdf", b"not-pdf"),
        ("fake.docx", b"PK-broken"),
        ("empty.txt", b"   \r\n"),
    ],
)
async def test_rejects_invalid_uploads_without_leaving_files(
    tmp_path,
    name,
    content,
):
    upload = UploadFile(filename=name, file=io.BytesIO(content))

    with pytest.raises(UploadRejected) as raised:
        await save_validated_upload(
            upload,
            tmp_path,
            max_upload_bytes=1024,
            max_extracted_bytes=4096,
        )

    assert raised.value.status_code == 400
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_rejects_upload_over_byte_limit(tmp_path):
    upload = UploadFile(filename="large.txt", file=io.BytesIO(b"12345"))

    with pytest.raises(UploadRejected) as raised:
        await save_validated_upload(
            upload,
            tmp_path,
            max_upload_bytes=4,
            max_extracted_bytes=4096,
        )

    assert raised.value.status_code == 413
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_rejects_docx_over_extracted_byte_limit(tmp_path):
    upload = UploadFile(filename="large.docx", file=io.BytesIO(make_docx("x" * 100)))

    with pytest.raises(UploadRejected) as raised:
        await save_validated_upload(
            upload,
            tmp_path,
            max_upload_bytes=4096,
            max_extracted_bytes=64,
        )

    assert raised.value.status_code == 413
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("report.pdf", b"%PDF-1.7\nvalid payload"),
        ("report.docx", make_docx()),
        ("notes.txt", "Xin chào".encode()),
    ],
)
async def test_accepts_supported_document_content(tmp_path, name, content):
    upload = UploadFile(filename=name, file=io.BytesIO(content))

    saved = await save_validated_upload(
        upload,
        tmp_path,
        max_upload_bytes=4096,
        max_extracted_bytes=4096,
    )

    assert saved.path.read_bytes() == content
    assert saved.path.name.endswith(f"_{name}")
    assert len(saved.path.name.split("_", 1)[0]) == 32
    assert saved.source_file == name


@pytest.mark.asyncio
async def test_same_content_has_same_document_id(tmp_path):
    first = UploadFile(filename="first.txt", file=io.BytesIO(b"hello"))
    second = UploadFile(filename="second.txt", file=io.BytesIO(b"hello"))

    one = await save_validated_upload(first, tmp_path, 1024, 4096)
    two = await save_validated_upload(second, tmp_path, 1024, 4096)

    assert one.document_id == two.document_id
