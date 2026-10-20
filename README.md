# 🤖 Multimodal RAG-Based AI Knowledge Assistant

A Retrieval-Augmented Generation (RAG) system that lets users upload documents, audio/video files, YouTube URLs and web pages, then ask questions in natural language and get answers grounded in that content, with the source files shown.

---

## 🎯 Features

- 📄 **Document Support** — PDF, CSV, Excel, Word documents
- 🎬 **Media Support** — Audio and video transcription via Faster-Whisper (uses the GPU when available)
- ▶️ **YouTube Integration** — Paste any YouTube URL and query its content
- 🌐 **Web Scraping** — Load any webpage directly into your knowledge base
- 🔍 **Hybrid Search** — Semantic vector search and BM25 keyword search merged with Reciprocal Rank Fusion
- 🛡️ **Grounded Answers** — The LLM answers only from retrieved content and says so when the documents don't contain the answer
- 📊 **Source Citations** — Every answer shows which files it came from
- 🧪 **Evaluation Pipeline** — Golden-dataset evaluation of retrieval quality and answer correctness/faithfulness

---

## 🛠️ Tech Stack

| Layer          | Technology                                                                  |
| -------------- | --------------------------------------------------------------------------- |
| Backend        | FastAPI, Python                                                             |
| LLM            | Any OpenRouter model via LangChain (configurable with `OPENROUTER_MODEL`) |
| Embeddings     | HuggingFace sentence-transformers (all-MiniLM-L6-v2)                        |
| Vector DB      | ChromaDB                                                                    |
| Keyword Search | BM25 (rank-bm25), cached in memory                                          |
| Database       | SQLite + SQLAlchemy                                                         |
| Transcription  | Faster-Whisper                                                              |
| Frontend       | Streamlit                                                                   |

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
│   ├── Dockerfile
│   ├── requirements.txt
│   └── requirements-render.txt
├── frontend/
│   ├── app.py
│   └── pages/
│       ├── 1_upload.py
│       └── 2_chat.py
├── evaluation/
│   ├── docs/                  # fixed test corpus (2 PDFs)
│   ├── golden_dataset.json    # 31 questions with reference answers
│   ├── run_eval.py            # evaluation script
│   └── results/               # saved results and grades
├── render.yaml
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

# With an NVIDIA GPU (CUDA 12.4 build of PyTorch)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install -r requirements-render.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your values:

```env
OPENROUTER_API_KEY=your-openrouter-api-key
OPENROUTER_MODEL=nvidia/nemotron-3.5-lightning:free
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

## 🔍 How It Works

```
Upload content (file / YouTube URL / webpage)
        ↓
Parse and extract text (Faster-Whisper for audio/video)
        ↓
Split into chunks (LangChain RecursiveCharacterTextSplitter)
        ↓
Generate embeddings (HuggingFace all-MiniLM-L6-v2)
        ↓
Store in ChromaDB with metadata
        ↓
User asks question
        ↓
Hybrid search: top 20 semantic + top 20 BM25 results,
merged with Reciprocal Rank Fusion → top 8 chunks
        ↓
Grounded prompt: answer only from the chunks,
or say the documents don't contain the answer
        ↓
Answer + source files returned
```

---

## 🧪 Evaluation

The `evaluation/` folder measures how well the system retrieves the right content and answers from it.

**Test set:** 31 questions over two research papers in `evaluation/docs/` ("Attention Is All You Need" and VL-JEPA):

- 26 answerable questions, each with a reference answer and an evidence phrase copied from the PDF
- 5 unanswerable questions (e.g. "Who won the 2022 FIFA World Cup?") that the system should decline

The script builds a separate, temporary index from these documents using the app's own parsing, chunking and storage code, so results don't depend on what you have uploaded.

**Metrics**

| Metric       | What it measures                                                                    |
| ------------ | ----------------------------------------------------------------------------------- |
| hit@k        | % of questions where a chunk containing the evidence phrase is in the top k results |
| MRR          | Mean of 1 / rank of the first correct chunk (1.0 = always first)                    |
| Correctness  | Answer matches the reference (correct = 1, partial = 0.5, incorrect = 0)            |
| Abstention   | % of unanswerable questions the system correctly declines                           |
| Faithfulness | % of answers whose every claim is supported by the retrieved chunks                 |

**Retrieval results** (26 answerable questions, top 8)

| Retriever              | hit@1           | hit@3           | hit@8           | MRR             |
| ---------------------- | --------------- | --------------- | --------------- | --------------- |
| Semantic only          | 30.8%           | 53.8%           | 69.2%           | 0.450           |
| BM25 only              | 57.7%           | 76.9%           | 92.3%           | 0.697           |
| **Hybrid (RRF)** | **57.7%** | **80.8%** | **96.2%** | **0.701** |

**Answer results** — answers generated by `nvidia/nemotron-3.5-lightning:free`, graded by manual review. The second run is after switching to the grounded prompt:

| Metric       | Original prompt | Grounded prompt |
| ------------ | --------------- | --------------- |
| Correctness  | 94.2%           | 90.4%           |
| Abstention   | 80%             | **100%**  |
| Faithfulness | 80.6%           | **100%**  |

The grounded prompt removed unsupported details and outside-knowledge answers entirely. The small drop in correctness came from two garbled responses from the free model, not from the model declining answerable questions.

These numbers come from a small test set (31 questions, 2 documents, one run each), so treat them as a before/after comparison rather than a precise quality score.

**Running the evaluation**

```bash
# From the project root, with backend dependencies installed and OPENROUTER_API_KEY set
python evaluation/run_eval.py                  # retrieval + answers, graded by an LLM judge
python evaluation/run_eval.py --no-judge       # retrieval + answers only, grade them later
python evaluation/run_eval.py --apply-grades evaluation/results/<results>.json <grades>.json
python evaluation/run_eval.py --ids att-01,neg-01   # run a subset of questions
python evaluation/run_eval.py --fresh          # ignore saved progress and start over
```

Progress is saved after every answer, so if the LLM provider times out, running the same command again resumes where it stopped. Set `EVAL_JUDGE_MODEL` to grade with a different model than the one being tested.

---

## 📊 Performance

Retrieval latency, measured on a 235-chunk knowledge base:

| Component             | Before | After   | Change                                                               |
| --------------------- | ------ | ------- | -------------------------------------------------------------------- |
| BM25 search           | ~50 ms | ~0.4 ms | Index cached in memory, rebuilt only when files are added or deleted |
| Full hybrid retrieval | ~85 ms | ~14 ms  | Duplicate semantic search removed and BM25 cached                    |

End-to-end response time is dominated by the LLM call and depends on the OpenRouter model. With the free Nemotron model, the median answer time in the evaluation was about 33 seconds.

---

## 📝 Environment Variables

| Variable               | Description                                                        |
| ---------------------- | ------------------------------------------------------------------ |
| `OPENROUTER_API_KEY` | OpenRouter API key (used by the LLM client)                        |
| `OPENROUTER_MODEL`   | OpenRouter model name                                              |
| `GEMINI_API_KEY`     | Google Gemini API key (legacy, currently unused)                   |
| `GEMINI_MODEL`       | Gemini model name (legacy, currently unused)                       |
| `EMBEDDING_MODEL`    | HuggingFace embedding model                                        |
| `DATABASE_URL`       | SQLite database URL                                                |
| `UPLOAD_DIR`         | Directory for uploaded files                                       |
| `CHROMA_PERSIST_DIR` | ChromaDB storage directory                                         |
| `MAX_FILE_SIZE_MB`   | Maximum upload size in MB                                          |
| `EVAL_JUDGE_MODEL`   | Optional: OpenRouter model used to grade answers in the evaluation |

---

## 📄 License

MIT License
