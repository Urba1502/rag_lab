# STEP 5 - The whole thing: Retrieve, Augment, Generate.
# Run:  python step5_rag.py

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"
COLLECTION = "workshop"
TOP_K = 3

# Set True to print the exact prompt. Do this at least once.
SHOW_PROMPT = False

# Two rules keep the model honest:
#   "ONLY the context" -> stops it answering from memory
#   "say I could not find" -> lets it admit when it does not know
ANSWER_TEMPLATE = """Answer the question using ONLY the context below.
If the context does not contain the answer, say
"I could not find that in the document." Do not use outside knowledge.
Cite the chunk id you used in square brackets.

Context:
{context}

Question: {question}

Answer:"""


def main():
    if not GROQ_API_KEY:
        print("GROQ_API_KEY not found. Check your .env file.")
        return

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)

    if vectorstore._collection.count() == 0:
        print("Vector store is empty - run step3_embed_store.py first.")
        return

    print("Ask about the document. Type 'quit' to exit.\n")

    while True:
        question = input("You > ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        # R : RETRIEVE
        results = vectorstore.similarity_search_with_score(question, k=TOP_K)
        print("\n  Retrieved:")
        for doc, distance in results:
            print(f"    - {doc.metadata['chunk_id']} "
                  f"(page {doc.metadata['page']}, distance {distance:.3f})")

        # A : AUGMENT
        # Glue the chunks together and drop them into the template.
        # That is the entire "augmentation" trick.
        context = "\n\n---\n\n".join(
            f"[{doc.metadata['chunk_id']}]\n{doc.page_content}"
            for doc, _ in results
        )
        prompt = ANSWER_TEMPLATE.format(context=context, question=question)

        if SHOW_PROMPT:
            print("\n  ----- EXACT PROMPT SENT TO THE MODEL -----")
            print(prompt)
            print("  ----- END OF PROMPT -----")

        # ---- G : GENERATE ----
        response = llm.invoke(prompt)
        print(f"\n  Answer: {response.content.strip()}\n")


if __name__ == "__main__":
    main()