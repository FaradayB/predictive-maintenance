"""
=============================================================================
 rag_pipeline.py
 Vehicle Predictive Maintenance — RAG Pipeline (Google Gemini)
=============================================================================
 Knowledge base: two SOP documents (Track 1 + Track 2)
   - sop_track1_technician_fault_diagnosis.md
   - sop_track2_owner_risk_alert.md

 Five mandatory stages (per rubric):
   1. Document Loading   — reads .md SOPs from docs/
   2. Chunking          — RecursiveCharacterTextSplitter, 500 chars / 50 overlap
   3. Embedding         — Google text-embedding-004
   4. Vector Store      — ChromaDB, persisted to chroma_db/
   5. Retrieval         — MMR search, top-k = 4

 Quick start:
   from predictivecare.rag import build_vectorstore, retrieve, format_context
   vs   = build_vectorstore()
   docs = retrieve("oil pressure low procedure", vs)
   ctx  = format_context(docs)

 Required .env:
   GOOGLE_API_KEY=your_key_here
=============================================================================
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Optional
from collections import Counter

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from predictivecare import config

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DOCS_DIR        = config.DOCS_DIR
CHROMA_DIR      = config.CHROMA_DIR
COLLECTION_NAME = "vehicle_maintenance_sop"

# Chunking strategy:
#   500-char chunks keep one SOP section (one fault class, one sensor row,
#   one alert template) together without splitting mid-table.
#   50-char overlap preserves the section heading at chunk boundaries so the
#   LLM knows which fault class an inspection step belongs to.
#   Separators are ordered for markdown structure: section headings first,
#   then paragraphs, then table rows (|), then lines.
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
TOP_K           = 4

EMBEDDING_MODEL = config.GOOGLE_EMBEDDING

# Map each SOP filename to a human-readable track label
SOP_TRACK_MAP = {
    "sop_track1_technician_fault_diagnosis.md": "Track 1 -- Technician SOP",
    "sop_track2_owner_risk_alert.md":           "Track 2 -- Owner Alert SOP",
}


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — DOCUMENT LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_documents(docs_dir: Path = DOCS_DIR) -> List[Document]:
    """
    Load SOP .md files from docs_dir.
    Tags each document with its track label and document type so the LLM
    can cite the source in its output.
    """
    if not docs_dir.exists():
        raise FileNotFoundError("directory not found")

    md_files = sorted(docs_dir.glob("*.md"))
    if not md_files:
        raise ValueError("No files found")

    log.info(f"[Stage 1] Loading SOP documents from '{docs_dir}' ...")

    docs: List[Document] = []
    for path in md_files:
        loader = TextLoader(str(path), encoding="utf-8")
        loaded = loader.load()
        track  = SOP_TRACK_MAP.get(path.name, "General SOP")
        for doc in loaded:
            doc.metadata["source"]   = path.name
            doc.metadata["track"]    = track
            doc.metadata["doc_type"] = "sop"
            doc.metadata["track_num"] = 1 if "track1" in path.name else (2 if "track2" in path.name else 0)
        docs.extend(loaded)
        log.info(
            f"   Loaded: {path.name}  "
            f"({len(loaded[0].page_content):,} chars)  [{track}]"
        )

    log.info(f"   Total documents loaded: {len(docs)}")
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Split SOP documents into overlapping chunks.

    Chunking strategy rationale:
    - chunk_size=500:   Keeps one SOP section (e.g. a single fault class with
                        its inspection steps) within a single chunk.
    - chunk_overlap=50: Preserves the section heading when a chunk starts
                        mid-section — critical for the LLM to know which
                        fault class an inspection step belongs to.
    - Separators ordered for markdown: section breaks (##/###) first, then
      paragraphs, then table rows (|), then line breaks, then spaces.
      This prevents sensor threshold tables from splitting mid-row.
    """
    log.info(
        f"[Stage 2] Chunking documents  "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ..."
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n## ", "\n### ", "\n\n", "\n", "|", " ", ""],
        add_start_index=True,
    )

    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    log.info(f"   Total chunks: {len(chunks)}")
    for src, n in Counter(c.metadata["source"] for c in chunks).most_common():
        track = SOP_TRACK_MAP.get(src, "General")
        log.info(f"   {src}: {n} chunks  [{track}]")

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Initialise Google Generative AI embedding model.
    Requires GOOGLE_API_KEY in .env or environment.

    The model is configured via GOOGLE_EMBEDDING (see config).
    - Supports task_type='retrieval_document' for indexing
      and task_type='retrieval_query' for queries
    """
    api_key = config.require_google_api_key()
    log.info(f"[Stage 3] Initializing embeddings  (model={EMBEDDING_MODEL}) ...")
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
        task_type="retrieval_document",
    )


def get_query_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Separate embedding instance for query-time retrieval.
    Google recommends different task_type for documents vs queries.
    """
    api_key = config.require_google_api_key()
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
        task_type="retrieval_query",
    )


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — VECTOR STORE
# ─────────────────────────────────────────────────────────────────────────────

def build_vectorstore(
    force_rebuild: bool = False,
    docs_dir: Path = DOCS_DIR,
    chroma_dir: Path = CHROMA_DIR,
) -> Chroma:
    """
    Build or reload the ChromaDB vector store from SOP documents.

    Args:
        force_rebuild: Re-embed all documents even if the store exists.
                       Set False in production to load instantly from disk.
        docs_dir:      Folder containing SOP .md files.
        chroma_dir:    ChromaDB persistence directory.

    Returns:
        Chroma vectorstore instance ready for retrieval.
    """
    doc_embeddings   = get_embeddings()
    query_embeddings = get_query_embeddings()

    if chroma_dir.exists() and not force_rebuild:
        log.info(f"[Stage 4] Loading vector store from '{chroma_dir}' ...")
        vs = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=query_embeddings,
            persist_directory=str(chroma_dir),
        )
        log.info(f"   Loaded {vs._collection.count()} vectors from disk.")
        return vs

    log.info(f"[Stage 4] Building vector store in '{chroma_dir}' ...")
    t0     = time.time()
    docs   = load_documents(docs_dir)
    chunks = chunk_documents(docs)

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=doc_embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(chroma_dir),
    )

    log.info(
        f"   Built: {vs._collection.count()} vectors  "
        f"({time.time() - t0:.1f}s)"
    )
    return vs


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def get_retriever(vectorstore: Chroma, k: int = TOP_K, track: Optional[int] = None):
    """
    Return an MMR retriever.

    MMR (Maximal Marginal Relevance) balances relevance and diversity.
    A pure similarity search on SOP content might return 4 nearly-identical
    chunks from the same threshold table. MMR ensures the result includes
    the threshold values, the inspection procedure, AND the recommended
    action — giving the LLM a complete picture to ground its output.

    lambda_mult=0.7 → 70% relevance, 30% diversity.
    fetch_k = k * 3 → retrieves 12 candidates, re-ranks to top 4.
    """
    log.info(f"[Stage 5] Retriever ready  (MMR, k={k}, track={track})")
    search_kwargs = {"k": k, "fetch_k": k * 3, "lambda_mult": 0.7}
    if track:
        search_kwargs["filter"] = {"track_num": track}
    return vectorstore.as_retriever(search_type="mmr", search_kwargs=search_kwargs)


def retrieve(
    query: str,
    vectorstore: Optional[Chroma] = None,
    k: int = TOP_K,
    track: Optional[int] = None,
) -> List[Document]:
    """
    Retrieve top-k SOP chunks relevant to a query.

    Args:
        query:       Natural-language query string.
        vectorstore: Pre-built Chroma instance. Loads from disk if None.
        k:           Number of chunks to return.

    Returns:
        List of Document objects with page_content and metadata.
    """

    if vectorstore is None:
        vectorstore = build_vectorstore(force_rebuild=False)

    retriever = get_retriever(vectorstore, k=k, track=track)
    docs      = retriever.invoke(query)

    log.info(f"Query '{query[:55]}' -> {len(docs)} chunks")
    for i, doc in enumerate(docs):
        log.info(
            f"   [{i+1}] {doc.metadata.get('track','?')}  "
            f"chunk {doc.metadata.get('chunk_id','?')}"
        )
    return docs


def format_context(docs: List[Document]) -> str:
    """
    Format retrieved SOP chunks into a single labelled context string
    ready to inject into the LLM system prompt.
    Each chunk is labelled with its track and source file for citation.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        track  = doc.metadata.get("track", "SOP")
        source = doc.metadata.get("source", "unknown")
        parts.append(
            f"[Context {i} | {track} | {source}]\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  RAG Pipeline — Self Test (Google Gemini, SOP Corpus)")
    print("=" * 60)

    vs = build_vectorstore(force_rebuild=True)

    test_queries = [
        # Track 1 — technician
        "oil pressure low inspection steps",
        "battery degradation alternator check",
        "engine misfire spark plug diagnosis",
        "cooling system overheating thermostat",
        # Track 2 — owner alert
        "high risk alert owner do not drive",
        "medium risk schedule service notification",
        "TPMS tyre pressure low owner warning",
        "Class 3 high risk immediate action",
    ]

    print(f"\nRunning {len(test_queries)} test queries ...\n")
    for q in test_queries:
        print(f"  Q: {q}")
        results = retrieve(q, vectorstore=vs, k=3)
        for doc in results:
            track   = doc.metadata.get("track", "?")
            chunk   = doc.metadata.get("chunk_id", "?")
            snippet = doc.page_content[:100].replace("\n", " ")
            print(f"    [{track} | chunk {chunk}]  {snippet}...")
        print()

    print("=" * 60)
    print("  Self-test complete. Vector store persisted to chroma_db/")
    print("=" * 60)
