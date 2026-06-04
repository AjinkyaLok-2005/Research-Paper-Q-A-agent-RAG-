# Research Paper Q&A Agent with Query Decomposition

A RAG system that lets you upload research paper PDFs and ask questions about them. Complex questions are automatically broken into sub-questions by a LangGraph agent, retrieved independently from ChromaDB, and synthesized into one cited answer.

---

## Tech Stack
- **LLM:** Groq API — llama-3.3-70b-versatile
- **Embeddings:** HuggingFace all-MiniLM-L6-v2 (local)
- **Vector Store:** ChromaDB (local, persistent)
- **Agent:** LangGraph (query decomposition)
- **PDF Parsing:** PyMuPDF
- **UI:** Streamlit

---

## Setup

```bash
git clone https://github.com/AjinkyaLok-2005/research-paper-qa.git
cd research-paper-qa
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at: https://console.groq.com

Run health check then launch:
```bash
python check_setup.py
streamlit run app.py
```

---

## How It Works

```
PDF Upload → chunk → embed → ChromaDB
Question  → LangGraph agent → sub-questions → retrieve → Groq LLM → cited answer
```

1. PDFs are split into 500-word chunks, embedded, and stored in ChromaDB
2. LangGraph agent classifies the question as simple or complex
3. Complex questions are decomposed into 2–4 focused sub-questions
4. Each sub-question independently retrieves top-4 chunks from ChromaDB
5. Groq LLM synthesizes one final answer with citations (paper name + page number)

---

## Project Structure

```
├── app.py              # Streamlit UI
├── check_setup.py      # Health check
├── requirements.txt
├── config/
│   └── settings.py     # All parameters
└── core/
    ├── ingestion.py    # PDF chunking
    ├── indexing.py     # Embedding + ChromaDB
    ├── agent.py        # LangGraph agent
    ├── retrieval.py    # Vector search
    └── synthesis.py    # Answer generation
```

---

## Author

**Ajinkya Lokhande** — [LinkedIn](https://www.linkedin.com/in/ajinkya-lokhande-142422295/) · [GitHub](https://github.com/AjinkyaLok-2005)