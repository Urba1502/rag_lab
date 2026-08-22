# STEP 2 - Load the PDF and cut it into chunks.
# Run:  python step2_chunk.py

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

FILE_PATH = Path("sample_docs/7_things.pdf")
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


def load_document(path: Path):
    """Read a file from disk and return it as a list of LangChain Documents."""
    if path.suffix.lower() == ".pdf":
        return PyPDFLoader(str(path)).load()
    return TextLoader(str(path), encoding="utf-8").load()


def main():
    # --- LOAD: for a PDF, this returns ONE Document per page ---
    docs = load_document(FILE_PATH)
    print(f"Loaded '{FILE_PATH.name}'")
    print(f"  Pages      : {len(docs)}")
    print(f"  Characters : {sum(len(d.page_content) for d in docs)}")

    # --- SPLIT ---
    # separators are tried IN ORDER: paragraph break, line break, space,
    # then raw characters. That is why chunks respect the document's shape.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"\nSplit into {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})\n")

    # --- LOOK AT THEM ---
    for i, chunk in enumerate(chunks):
        preview = " ".join(chunk.page_content.split())[:80]
        page = chunk.metadata["page"] + 1        # PyPDF counts pages from 0
        print(f"CHUNK {i} | page {page} | {len(chunk.page_content)} chars")
        print(f"  {preview}...")


if __name__ == "__main__":
    main()