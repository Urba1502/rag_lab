# STEP 6 - Put the RAG pipeline behind a web API.
# Run:  uvicorn step6_api:app --port 8001
# Then open:  http://127.0.0.1:8001/docs

import os
import shutil
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"
COLLECTION = "workshop"
DATA_DIR = Path("./data")
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

ANSWER_TEMPLATE = """Answer the question using ONLY the context below.
If the context does not contain the answer, say
"I could not find that in the document." Do not use outside knowledge.
Cite the chunk id you used in square brackets.

Context:
{context}

Question: {question}

Answer:"""

app = FastAPI(title="AI Workshop RAG API", version="1.0.0")

# Loaded ONCE at startup. Loading it per request would add seconds to every call.
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vectorstore() -> Chroma:
    return Chroma(collection_name=COLLECTION, persist_directory=CHROMA_DIR,
                  embedding_function=embeddings)


def get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set")
    return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


class SourceChunk(BaseModel):
    chunk_id: str
    page: int
    distance: float
    content: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


class UploadResponse(BaseModel):
    filename: str
    pages: int
    chunks_added: int


@app.get("/health")
def health():
    return {"status": "ok", "chunks": get_vectorstore()._collection.count()}


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    """Upload a PDF or text file: it is chunked and stored automatically."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Keep only base name. The client controls this value, and a name like
    # "../../step6_api.py" would otherwise overwrite our own source code.
    safe_name = Path(file.filename).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in [".pdf", ".txt", ".md", ".log"]:
        raise HTTPException(status_code=400, detail=f"Unsupported type: {suffix}")

    dest = DATA_DIR / safe_name
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)

    if suffix == ".pdf":
        docs = PyPDFLoader(str(dest)).load()
    else:
        docs = TextLoader(str(dest), encoding="utf-8").load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise HTTPException(status_code=400,
                            detail="No text found. A scanned PDF needs OCR.")

    stem = Path(safe_name).stem
    for i, chunk in enumerate(chunks):
        chunk.metadata = {
            "chunk_id": f"{stem}-chunk-{i}",
            "source": safe_name,
            "page": chunk.metadata.get("page", 0) + 1,
        }

    get_vectorstore().add_texts(
        texts=[c.page_content for c in chunks],
        metadatas=[c.metadata for c in chunks],
        ids=[c.metadata["chunk_id"] for c in chunks],
    )
    return UploadResponse(filename=safe_name, pages=len(docs),
                          chunks_added=len(chunks))


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest):
    """Retrieve -> Augment -> Generate, exactly like step5_rag.py."""
    vectorstore = get_vectorstore()
    if vectorstore._collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents. Upload one first.")

    results = vectorstore.similarity_search_with_score(payload.query, k=payload.top_k)

    context = "\n\n---\n\n".join(
        f"[{doc.metadata['chunk_id']}]\n{doc.page_content}" for doc, _ in results
    )
    prompt = ANSWER_TEMPLATE.format(context=context, question=payload.query)
    answer = get_llm().invoke(prompt).content.strip()

    return QueryResponse(
        answer=answer,
        sources=[
            SourceChunk(chunk_id=doc.metadata["chunk_id"],
                        page=doc.metadata["page"],
                        distance=float(distance),
                        content=doc.page_content[:300])
            for doc, distance in results
        ],
    )