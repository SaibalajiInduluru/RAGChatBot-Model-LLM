# 🧠 RAG-Based PDF Q&A Chatbot (Local LLM)

This is a **Retrieval-Augmented Generation (RAG)** chatbot that allows users to **ask questions about the contents of a PDF** using **local LLM inference** via [Ollama](https://ollama.com/). It uses:
- `PyPDF2` for PDF reading
- `sentence-transformers` for embedding
- `ChromaDB` for vector storage and similarity search
- `Ollama` for LLM-based local response generation

---

## 🚀 Features

- 📄 Load and process any PDF file
- ✂️ Automatically chunks and embeds the document
- 🔍 Retrieves top relevant chunks using semantic similarity
- 💬 Generates contextual answers using a locally running LLM model (e.g., LLaMA 3 via Ollama)

---

## 🛠️ Requirements

Install the following packages (you can use `pip install -r requirements.txt`):

```txt
PyPDF2
chromadb
sentence-transformers
requests
