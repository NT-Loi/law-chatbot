"""
PostgreSQL Database Models for Law Chatbot
Using SQLModel (SQLAlchemy + Pydantic)
"""
from sqlmodel import Field, SQLModel, Relationship, create_engine, UniqueConstraint
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Text, JSON

import os
from dotenv import load_dotenv
from typing import Optional, List
from datetime import datetime

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
DB_ECHO = os.getenv("DB_ECHO", "false").lower() == "true"

# Sync engine for migrations/ingest
engine = create_engine(DATABASE_URL, echo=DB_ECHO)

# Async engine for API
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
async_engine = create_async_engine(ASYNC_DATABASE_URL, echo=DB_ECHO)
async_session_factory = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


# ============ VBQPPL Models ============

class VBQPPLDoc(SQLModel, table=True):
    """
    Văn bản quy phạm pháp luật - Document level
    Stores full document metadata and content
    """
    __tablename__ = "vbqppl_docs"
    
    id: str = Field(primary_key=True)  # e.g. "15/2012/TT-BGTVT"
    title: Optional[str] = Field(default=None, index=True)
    url: Optional[str] = Field(default=None)
    content: Optional[str] = Field(default=None, sa_column=Column(Text))  # Full document content
    status: Optional[str] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    original_name: Optional[str] = Field(default=None)
    original_link: Optional[str] = Field(default=None)
    
    # Relationships
    sections: List["VBQPPLSection"] = Relationship(back_populates="doc", sa_relationship_kwargs={"cascade": "all, delete-orphan"})


class VBQPPLSection(SQLModel, table=True):
    """
    VBQPPL Section - Individual article/section within a document
    Used for quick retrieval when displaying references
    """
    __tablename__ = "vbqppl_sections"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    hash_id: Optional[str] = Field(default=None, index=True)  # MD5 hash of doc_id + hierarchy_path (matches Qdrant ID)
    doc_id: str = Field(foreign_key="vbqppl_docs.id", index=True)
    label: str = Field(index=True)  # e.g. "Điều 1. Phạm vi điều chỉnh"
    content: str = Field(sa_column=Column(Text))
    hierarchy_path: Optional[str] = Field(default=None)
    section_type: Optional[str] = Field(default=None)  # "dieu", "chuong", etc.
    
    # Relationship back to parent doc
    doc: VBQPPLDoc = Relationship(back_populates="sections")


# ============ Pháp Điển Models ============

class PhapDienDieu(SQLModel, table=True):
    """
    Pháp Điển - Điều level
    Each "Điều" from the Pháp Điển system
    """
    __tablename__ = "phapdien_dieu"
    
    id: str = Field(primary_key=True)  # UUID from source
    chi_muc: Optional[str] = Field(default=None)
    mapc: Optional[str] = Field(default=None, index=True)  # Mã phân cấp
    ten: str = Field(index=True)  # e.g. "Điều 36.3.LQ.1. Thanh niên"
    noi_dung: str = Field(sa_column=Column(Text))  # Content
    
    # Hierarchical references
    chu_de_id: Optional[str] = Field(default=None, index=True)
    de_muc_id: Optional[str] = Field(default=None, index=True)
    chuong_mapc: Optional[str] = Field(default=None)
    stt: Optional[int] = Field(default=None)
    
    # VBQPPL references stored as JSON array
    vbqppl_refs: Optional[str] = Field(default=None, sa_column=Column(Text))  # JSON string of references


# ============ Chat Session Models ============

class ChatSession(SQLModel, table=True):
    """Chat session for storing conversation history"""
    __tablename__ = "chat_sessions"
    
    id: str = Field(primary_key=True)
    title: str = Field(index=True)
    created_at: datetime = Field(default_factory=datetime.now)
    
    messages: List["ChatMessage"] = Relationship(
        back_populates="session", 
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )


class ChatMessage(SQLModel, table=True):
    """Individual chat message"""
    __tablename__ = "chat_messages"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="chat_sessions.id", index=True)
    role: str  # 'user' or 'assistant'
    content: str = Field(sa_column=Column(Text))
    sources: Optional[str] = Field(default=None, sa_column=Column(Text))  # JSON-stringified sources
    created_at: datetime = Field(default_factory=datetime.now)
    
    session: ChatSession = Relationship(back_populates="messages")


# ============ Database Utilities ============

def init_db(drop_all: bool = False):
    """Initialize database tables"""
    if drop_all:
        print("⚠️  Dropping all tables...")
        SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
    print("✅ Database tables created successfully!")


async def get_async_session():
    """Async session generator for FastAPI dependency injection"""
    async with async_session_factory() as session:
        yield session


if __name__ == "__main__":
    # Run directly to create tables
    import sys
    drop = "--drop" in sys.argv
    init_db(drop_all=drop)
