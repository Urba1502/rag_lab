# STEP 4 - Semantic search. No LLM here - just the vector database.
# Run:  python step4_search.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"
COLLECTION = "workshop"
TOP_K = 3


def main():
    # The SAME embedding model used in Step 3. A different model would put the
    # question in a different vector space, and distances would be meaningless.
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma(
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    count = vectorstore._collection.count()
    print(f"Collection holds {count} chunks.")
    if count == 0:
        print("EMPTY - run step3_embed_store.py first.")
        return

    print("Type a question. Type 'quit' to exit.\n")

    while True:
        query = input("Search > ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        # score is a DISTANCE: smaller = closer = better match
        results = vectorstore.similarity_search_with_score(query, k=TOP_K)

        print()
        for rank, (doc, distance) in enumerate(results, start=1):
            text = " ".join(doc.page_content.split())
            print(f"  #{rank}  distance={distance:.4f}  "
                  f"{doc.metadata['chunk_id']}  (page {doc.metadata['page']})")
            print(f"      {text[:95]}...")
        print()


if __name__ == "__main__":
    main()