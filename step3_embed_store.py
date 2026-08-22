# STEP 3 - Turn the chunks into vectors and store them in ChromaDB.
# Run:  python step3_embed_store.py

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

FILE_PATH = Path("sample_docs/7_things.pdf")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"
COLLECTION = "workshop"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


def load_and_chunk(path: Path):
    """Step 2, packed into one function."""
    if path.suffix.lower() == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    else:
        docs = TextLoader(str(path), encoding="utf-8").load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def main():
    # --- 1. THE EMBEDDING MODEL (runs locally, no API key, free) ---
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # See what an embedding actually is
    vector = embeddings.embed_query("staying disciplined")
    print(f"  One sentence -> {len(vector)} numbers")
    print(f"  First 8: {[round(x, 4) for x in vector[:8]]}")

    # --- 2. CHUNK ---
    chunks = load_and_chunk(FILE_PATH)
    print(f"\nDocument split into {len(chunks)} chunks")

    # --- 3. CLEAN METADATA ---
    # A PDF arrives with a dozen fields (producer, trapped, ...). We keep only
    # what we need. This also avoids a crash: Chroma rejects None values, and
    # many real PDFs have empty fields.
    for i, chunk in enumerate(chunks):
        chunk.metadata = {
            "chunk_id": f"{FILE_PATH.stem}-chunk-{i}",
            "source": FILE_PATH.name,
            "page": chunk.metadata.get("page", 0) + 1,
        }

    # --- 4. STORE ---
    vectorstore = Chroma(
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,      # delete this line -> memory only
        embedding_function=embeddings,
    )
    vectorstore.add_texts(
        texts=[c.page_content for c in chunks],
        metadatas=[c.metadata for c in chunks],
        ids=[c.metadata["chunk_id"] for c in chunks],   # stable ids = upsert
    )

    print(f"Stored. Total items in collection: {vectorstore._collection.count()}")
    print(f"Saved to disk: {CHROMA_DIR}/")


if __name__ == "__main__":
    main()