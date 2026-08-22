# STEP 1 - A working chatbot in 20 lines.
# Run:  python step1_chatbot.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def main():
    if not GROQ_API_KEY:
        print("GROQ_API_KEY not found. Check your .env file.")
        return

    # temperature=0 means "give me the most likely answer", not a creative one.
    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)

    print(f"Chatbot ready (model: {GROQ_MODEL}). Type 'quit' to exit.\n")

    while True:
        question = input("You > ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        # One line. This is the entire chatbot.
        response = llm.invoke(question)

        print(f"\nBot > {response.content.strip()}\n")


if __name__ == "__main__":
    main()