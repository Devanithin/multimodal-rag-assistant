from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime, timezone
from models.base import Base


class File(Base):
    __tablename__ = "files"

    # ── Primary Key ──
    id = Column(Integer, primary_key=True, index=True)

    # ── File Info ──
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # pdf, image, csv, audio, video
    size = Column(Integer, nullable=False)       # size in bytes

    # ── Storage ──
    file_path = Column(String, nullable=False)   # where file is saved on disk

    # ── Processing ──
    chunk_count = Column(Integer, default=0)     # how many chunks were created

    # ── Timestamps ──
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))