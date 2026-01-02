from typing import Optional, List
from sqlmodel import Field, SQLModel, Relationship, create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from config import DB_ECHO

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

engine = create_engine(DATABASE_URL, echo=DB_ECHO)
async_engine = create_async_engine(ASYNC_DATABASE_URL, echo=DB_ECHO)
async_session_factory = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

class PhapDienNode(SQLModel, table=True):
    __tablename__ = "phapdien_nodes"
    id: str = Field(primary_key=True) 
    content: str
    title: str = Field(index=True)
    demuc_id: Optional[str] = Field(default=None, index=True)
    # label: Optional[str] = None
    
    references: List["PhapDienReference"] = Relationship(back_populates="phapdien_node")


class VBQPPLDoc(SQLModel, table=True):
    __tablename__ = "vbqppl_docs"
    id: str = Field(primary_key=True)
    title: Optional[str] = Field(default=None, index=True)
    url: str
    is_crawled: bool = Field(default=False)
    
    nodes: List["VBQPPLNode"] = Relationship(back_populates="doc")


class VBQPPLNode(SQLModel, table=True):
    __tablename__ = "vbqppl_nodes"
    id: Optional[int] = Field(default=None, primary_key=True) # MAPC
    doc_id: str = Field(foreign_key="vbqppl_docs.id", index=True)
    # anchor: str = Field(index=True)
    # type: str = Field(index=True)
    label: str = Field(index=True)
    # title: Optional[str] = None
    content: str
    # parent_id: Optional[int] = Field(default=None, foreign_key="vbqppl_nodes.id")
    is_structure: bool = Field(default=True)

    doc: VBQPPLDoc = Relationship(back_populates="nodes")


class PhapDienReference(SQLModel, table=True):
    __tablename__ = "phapdien_references"
    id: Optional[int] = Field(default=None, primary_key=True)
    phapdien_id: str = Field(foreign_key="phapdien_nodes.id", index=True)
    vbqppl_doc_id: str = Field(foreign_key="vbqppl_docs.id")
    vbqppl_label: str = Field(index=True)
    details: str

    phapdien_node: PhapDienNode = Relationship(back_populates="references")


class PhapDienRelation(SQLModel, table=True):
    __tablename__ = "phapdien_relations"
    id: Optional[int] = Field(default=None, primary_key=True)
    source_id: str = Field(foreign_key="phapdien_nodes.id", index=True)
    target_id: str = Field(foreign_key="phapdien_nodes.id", index=True)


def init_db(drop_all: bool = False):
    if drop_all:
        print("Dropping all tables...")
        SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)


async def get_async_session():
    async with async_session_factory() as session:
        yield session

if __name__ == "__main__":
    init_db()
    print("Database tables created successfully!")
