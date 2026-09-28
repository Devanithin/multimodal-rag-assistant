# 🤖 Multimodal RAG-Based AI Knowledge Assistant

A production-grade Retrieval-Augmented Generation (RAG) system that allows users to upload documents, audio/video files, YouTube URLs, and web pages — then query them using natural language with cited answers.

---

## 🎯 Features

- 📄 **Document Support** — PDF, CSV, Excel, Word documents
- 🎬 **Media Support** — Audio and video transcription via Faster-Whisper (GPU accelerated)
- ▶️ **YouTube Integration** — Paste any YouTube URL and query its content
- 🌐 **Web Scraping** — Load any webpage directly into your knowledge base
- 🔍 **Hybrid Search** — Combines semantic vector search and BM25 keyword search
- 💬 **Natural Language Chat** — Ask questions and get cited answers
- ⚡ **Optimized Latency** — 61.8% reduction (12s → 4.5s) via LRU caching
- 📊 **Source Citations** — Every answer shows which file it came from

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python |
| LLM | OpenRouter (Gemini 2.0 Flash) via LangChain |
| Embeddings | HuggingFace sentence-transformers |
| Vector DB | ChromaDB |
| Keyword Search | BM25 (rank-bm25) |
| Database | SQLite + SQLAlchemy |
| Transcription | Faster-Whisper |
| Frontend | Streamlit |

---

## 📁 Project Structure
```
multimodal-rag-assistant/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── dependencies.py
│   ├── api/
│   │   ├── files.py
│   │   └── query.py
│   ├── core/
│   │   ├── ingestion/
│   │   │   ├── pdf_parser.py
│   │   │   ├── csv_parser.py
│   │   │   ├── docs_parser.py
│   │   │   ├── media_processor.py
│   │   │   ├── youtube_loader.py
│   │   │   ├── web_loader.py
│   │   │   └── chunker.py
│   │   ├── retrieval/
│   │   │   ├── vector_store.py
│   │   │   ├── bm25_search.py
│   │   │   └── hybrid_search.py
│   │   ├── embeddings/
│   │   │   └── text_embedder.py
│   │   └── llm/
│   │       └── llm_client.py
│   ├── models/
│   │   ├── base.py
│   │   └── file.py
│   └── requirements.txt
├── frontend/
│   ├── app.py
│   └── pages/
│       ├── 1_upload.py
│       └── 2_chat.py
├── evaluation/
├── infra/
└── .gitignore
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Devanithin/multimodal-rag-assistant.git
cd multimodal-rag-assistant
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
```

Edit `.env` and add your values:
```env
OPENROUTER_API_KEY=your-openrouter-api-key
OPENROUTER_MODEL=google/gemini-2.0-flash-exp:free
```

Get your free OpenRouter API key from: https://openrouter.ai/keys

### 5. Run the backend
```bash
cd backend
uvicorn main:app --reload
```

### 6. Run the frontend
```bash
cd frontend
streamlit run app.py
```

---

## 🚀 Usage

1. Open `http://localhost:8501` in your browser
2. Go to **Upload** page and add your content
3. Go to **Chat** page and ask questions
4. Get answers with source citations and latency metrics

---

## 📊 Performance

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Retrieval Time | 4.253s | 0.011s | 99.7% faster |
| Total Latency | 12.02s | 4.581s | 61.8% faster |

Optimization achieved via Python `lru_cache` on embedding model, vector store, and LLM client initialization.

---

## 🔍 How It Works
```
Upload content (file / YouTube URL / webpage)
        ↓
Parse and extract text
        ↓
Split into chunks (LangChain RecursiveCharacterTextSplitter)
        ↓
Generate embeddings (HuggingFace all-MiniLM-L6-v2)
        ↓
Store in ChromaDB with metadata
        ↓
User asks question
        ↓
Hybrid search (Semantic + BM25)
        ↓
Top chunks sent to LLM (OpenRouter)
        ↓
Answer + source citations returned
```

---

## 📝 Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter API key (used by the LLM client) |
| `OPENROUTER_MODEL` | OpenRouter model name |
| `GEMINI_API_KEY` | Google Gemini API key (legacy, currently unused) |
| `GEMINI_MODEL` | Gemini model name (legacy, currently unused) |
| `EMBEDDING_MODEL` | HuggingFace embedding model |
| `DATABASE_URL` | SQLite database URL |
| `UPLOAD_DIR` | Directory for uploaded files |
| `CHROMA_PERSIST_DIR` | ChromaDB storage directory |
| `MAX_FILE_SIZE_MB` | Maximum upload size in MB |

---

## 📄 License

MIT License