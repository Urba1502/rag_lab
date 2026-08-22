# RAG Lab

A simple **Retrieval-Augmented Generation (RAG)** project built with Python.

This project demonstrates the basic RAG pipeline: loading a document, splitting it into chunks, converting the chunks into embeddings, storing them in a vector database, and retrieving the most relevant chunks when a user asks a question.

## How RAG Works

```text
Document
   ↓
Text Splitting
   ↓
Chunks
   ↓
Embeddings
   ↓
Vector Store
   ↓
User Question
   ↓
Similarity Search
   ↓
Relevant Chunks
```

## Features

* Load PDF/text documents
* Split documents into smaller chunks
* Generate embeddings using Hugging Face
* Store embeddings in a vector database
* Perform similarity search
* Retrieve relevant document chunks for a question
* Use environment variables for API configuration

## Technologies Used

* **Python**
* **LangChain**
* **Hugging Face Embeddings**
* **Chroma**
* **Groq**
* **python-dotenv**

## Project Structure

```text
rag_lab/
│
├── venv/
│
├── documents/
│   └── your_document.pdf
│
├── step1_load.py
├── step2_split.py
├── step3_embed.py
├── step4_search.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd rag_lab
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=your_model_name
```

Do **not** commit your `.env` file to GitHub.

Make sure `.env` is included in `.gitignore`:

```text
.env
venv/
__pycache__/
```

## Running the Project

Run the scripts in order according to the current pipeline.

For example:

```bash
python step1_load.py
python step2_split.py
python step3_embed.py
python step4_search.py
```

The search step allows you to enter questions and retrieves the most relevant chunks from the stored document.

Example:

```text
Collection holds 8 chunks.

Search > What is retrieval augmented generation?
```

The system then performs a similarity search and returns the relevant document chunks.

## Current RAG Pipeline

The current project focuses on the fundamental RAG workflow:

1. **Load** the source document.
2. **Split** the document into smaller chunks.
3. **Embed** each chunk into a numerical vector.
4. **Store** the vectors in a vector database.
5. **Embed the user's question.**
6. **Search** for the most similar chunks.
7. **Retrieve** the relevant information.

This project is intentionally kept simple to understand the core concepts before moving toward more advanced RAG architectures.

## Future Improvements

Possible improvements include:

* Add an LLM response-generation step
* Build a complete question-answering pipeline
* Improve chunking strategies
* Add metadata filtering
* Experiment with different embedding models
* Add reranking
* Build a simple web interface
* Experiment with advanced RAG techniques such as:

  * HyDE RAG
  * Fusion RAG
  * Corrective RAG
  * Self-RAG
  * GraphRAG
  * Agentic RAG

## Goal

The goal of this project is to understand the fundamentals of **Retrieval-Augmented Generation** by building the system step by step rather than relying on a high-level framework without understanding what happens underneath.

---

**Status:** Learning Project

**Author:** Urba1502
