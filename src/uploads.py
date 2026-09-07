import hashlib
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

READ_SIZE = 1024 * 1024
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@dataclass(frozen=True)
class SavedUpload:
    path: Path
    document_id: str
    source_file: str


class UploadRejected(ValueError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _validate_pdf(path: Path) -> None:
    if not path.read_bytes()[:5] == b"%PDF-":
        raise UploadRejected(400, "The uploaded PDF is invalid.")


def _validate_docx(path: Path, max_extracted_bytes: int) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = {member.filename for member in members}
            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                raise UploadRejected(400, "The uploaded DOCX is invalid.")
            if sum(member.file_size for member in members) > max_extracted_bytes:
                raise UploadRejected(413, "The extracted document is too large.")
    except zipfile.BadZipFile as exc:
        raise UploadRejected(400, "The uploaded DOCX is invalid.") from exc


def _validate_text(path: Path) -> None:
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise UploadRejected(400, "The uploaded text file must be UTF-8.") from exc
    if not content.strip():
        raise UploadRejected(400, "The uploaded text file is empty.")


def _validate_content(path: Path, suffix: str, max_extracted_bytes: int) -> None:
    if suffix == ".pdf":
        _validate_pdf(path)
    elif suffix == ".docx":
        _validate_docx(path, max_extracted_bytes)
    elif suffix == ".txt":
        _validate_text(path)
    else:
        raise UploadRejected(400, "Only PDF, DOCX, and TXT files are supported.")


async def save_validated_upload(
    file: UploadFile,
    root: Path,
    max_upload_bytes: int,
    max_extracted_bytes: int,
) -> SavedUpload:
    if not file.filename:
        raise UploadRejected(400, "A file is required.")

    source_file = Path(file.filename).name
    suffix = Path(source_file).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UploadRejected(400, "Only PDF, DOCX, and TXT files are supported.")

    root.mkdir(parents=True, exist_ok=True)
    temporary_path = root / f".{uuid.uuid4()}.upload"
    digest = hashlib.sha256()
    total_bytes = 0

    try:
        with temporary_path.open("wb") as output:
            while chunk := await file.read(READ_SIZE):
                total_bytes += len(chunk)
                if total_bytes > max_upload_bytes:
                    raise UploadRejected(413, "The uploaded document is too large.")
                output.write(chunk)
                digest.update(chunk)

        if total_bytes == 0:
            raise UploadRejected(400, "The uploaded document is empty.")

        _validate_content(temporary_path, suffix, max_extracted_bytes)
        document_id = digest.hexdigest()
        saved_path = root / f"{document_id}_{source_file}"
        temporary_path.replace(saved_path)
        return SavedUpload(
            path=saved_path,
            document_id=document_id,
            source_file=source_file,
        )
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
