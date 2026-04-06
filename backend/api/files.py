from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from pydantic import BaseModel
import os

from dependencies import get_db
from models.file import File as FileModel
from config import settings
from core.ingestion.pdf_parser import parse_pdf
from core.ingestion.csv_parser import parse_csv
from core.ingestion.docs_parser import parse_docx
from core.ingestion.media_processor import parse_media
from core.ingestion.web_loader import parse_web_url
from core.ingestion.youtube_loader import parse_youtube_url
from core.ingestion.chunker import chunk_text
from core.retrieval.vector_store import store_chunks, delete_file_chunks

router = APIRouter(prefix="/files", tags=["files"])

PARSERS = {
    "pdf": parse_pdf,
    "csv": parse_csv,
    "xlsx": parse_csv,
    "docx": parse_docx,
    "mp3": parse_media,
    "mp4": parse_media,
    "wav": parse_media,
    "mov": parse_media,
    "m4a": parse_media,
}


# ── Upload File ──
@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    contents = await file.read()
    size = len(contents)
    max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024

    if size > max_size:
        raise HTTPException(status_code=400, detail=f"File too large. Max size is {settings.MAX_FILE_SIZE_MB}MB")

    extension = os.path.splitext(file.filename)[1].lower().strip(".")
    if extension not in PARSERS:
        raise HTTPException(status_code=400, detail="File type not supported")

    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)

    parse_fn = PARSERS[extension]
    text = parse_fn(file_path)
    chunks = chunk_text(text)

    new_file = FileModel(
        filename=file.filename,
        original_filename=file.filename,
        file_type=extension,
        size=size,
        file_path=file_path,
        chunk_count=len(chunks)
    )
    db.add(new_file)
    await db.commit()
    await db.refresh(new_file)

    store_chunks(chunks, file_id=new_file.id, filename=file.filename)

    return {
        "id": new_file.id,
        "filename": new_file.filename,
        "file_type": new_file.file_type,
        "size": new_file.size,
        "chunk_count": new_file.chunk_count,
        "uploaded_at": new_file.uploaded_at
    }


# ── Upload from URL ──
class URLRequest(BaseModel):
    url: str
    name: str = ""


@router.post("/upload-url")
async def upload_from_url(request: URLRequest, db: AsyncSession = Depends(get_db)):
    url = request.url.strip()

    if "youtube.com" in url or "youtu.be" in url:
        source_type = "youtube"
        text = parse_youtube_url(url)
    else:
        source_type = "web"
        text = parse_web_url(url)

    if not text:
        raise HTTPException(status_code=400, detail="Could not extract content from URL")

    name = request.name or url
    chunks = chunk_text(text)

    new_file = FileModel(
        filename=name,
        original_filename=url,
        file_type=source_type,
        size=len(text),
        file_path=url,
        chunk_count=len(chunks)
    )
    db.add(new_file)
    await db.commit()
    await db.refresh(new_file)

    store_chunks(chunks, file_id=new_file.id, filename=name)

    return {
        "id": new_file.id,
        "filename": name,
        "file_type": source_type,
        "size": len(text),
        "chunk_count": len(chunks)
    }


# ── List Files ──
@router.get("/")
async def list_files(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(FileModel))
    files = result.scalars().all()
    return [
        {
            "id": f.id,
            "filename": f.filename,
            "file_type": f.file_type,
            "size": f.size,
            "chunk_count": f.chunk_count,
            "uploaded_at": f.uploaded_at
        }
        for f in files
    ]


# ── Delete File ──
@router.delete("/{file_id}")
async def delete_file(file_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(FileModel).where(FileModel.id == file_id))
    file = result.scalar_one_or_none()

    if not file:
        raise HTTPException(status_code=404, detail="File not found")

    if os.path.exists(file.file_path):
        os.remove(file.file_path)

    delete_file_chunks(file_id)

    await db.execute(delete(FileModel).where(FileModel.id == file_id))
    await db.commit()

    return {"message": f"{file.filename} deleted successfully"}