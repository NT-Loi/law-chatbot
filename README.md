# Vietnamese Law RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for Vietnamese legal documents, built with LangChain, Qdrant, PostgreSQL, and a modern React frontend.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                        │
└─────────────────────────────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (app.py)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Router    │  │  RAG Chain  │  │    Chat Strategies      │  │
│  │  (chat.py)  │  │  (rag.py)   │  │  Legal/Web/Hybrid/Chit  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         ▼                   ▼                      ▼
┌─────────────┐     ┌─────────────┐      ┌──────────────────┐
│   Qdrant    │     │ PostgreSQL  │      │   Voyage AI      │
│  (Vectors)  │     │ (Documents) │      │   (Reranker)     │
└─────────────┘     └─────────────┘      └──────────────────┘
```

## 🧠 RAG Pipeline Details

### Document Chunking Strategy

The system uses **hierarchical section-based chunking**. Each "Điều" (Article) is the primary chunk unit with its full hierarchy path preserved:

```
VBQPPL Document
├── Phần (Part)
│   ├── Chương (Chapter)
│   │   ├── Mục (Section)
│   │   │   └── Điều (Article) ← Primary chunk unit
```

**Why Điều-level chunking?** Legal articles are semantic units; splitting mid-article loses meaning. Each chunk is embedded with context: `{title}\n{hierarchy_path}\n{content}`.

### Complete RAG Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Query + History                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         1. ChatRouter (LLM)                                 │
│                     Classify: LEGAL / NON_LEGAL                             │
│                     (Uses last 2 messages for context)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                            │                    │
                         LEGAL               NON_LEGAL
                            │                    │
                            ▼                    ▼
┌────────────────────────────────────┐    ┌─────────────────┐
│        LegalRAGChain               │    │  ChitChatChain  │
│                                    │    │  (Direct LLM)   │
│ ┌────────────────────────────────┐ │    └─────────────────┘
│ │ 2. Multi-Query Reflection      │ │
│ │    LLM generates 3 queries:    │ │
│ │    Q1: Contextualized          │ │
│ │    Q2: Legal terminology       │ │
│ │    Q3: Broader domain terms    │ │
│ │    → Q1 used for reranking     │ │
│ └────────────────────────────────┘ │
│                 │                  │
│                 ▼                  │
│ ┌────────────────────────────────┐ │
│ │ 3. Parallel Hybrid Search      │ │
│ │    3 queries × (Dense + BM25)  │ │
│ │    Qdrant RRF Fusion           │ │
│ │    → 30-60 candidates          │ │
│ └────────────────────────────────┘ │
│                 │                  │
│                 ▼                  │
│ ┌────────────────────────────────┐ │
│ │ 4. Voyage AI Rerank            │ │
│ │    Query: Q1 (contextualized)  │ │
│ │    → Top 20 scored             │ │
│ └────────────────────────────────┘ │
│                 │                  │
│                 ▼                  │
│ ┌────────────────────────────────┐ │
│ │ 5. Confidence-Based Filtering  │ │
│ │    Score > 0.75? Use top docs  │ │
│ │    Score ≤ 0.75? LLM selects   │ │
│ └────────────────────────────────┘ │
│                 │                  │
│                 ▼                  │
│ ┌────────────────────────────────┐ │
│ │ 6. Answer Generation (LLM)     │ │
│ │    Context + Last 4 messages   │ │
│ │    → Streaming response        │ │
│ │    → <USED_DOCS> citations     │ │
│ └────────────────────────────────┘ │
└────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Streaming Response + Rich Citations                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Stage Details

| Stage | Component | Description |
|-------|-----------|-------------|
| **1. Router** | `ChatRouter` | Classifies query as LEGAL/NON_LEGAL using last 2 history messages |
| **2. Reflection** | `REFLECTION_SYSTEM_PROMPT` | Generates 3 search queries; Q1 resolves pronouns from history |
| **3. Search** | `rag.retrieve()` | Hybrid search (dense + BM25) with RRF fusion, runs 3 queries in parallel |
| **4. Rerank** | Voyage AI `rerank-2` | Semantic reranking using contextualized Q1 |
| **5. Filter** | Confidence check | Score > 0.75 skips LLM; otherwise LLM selects relevant doc IDs |
| **6. Answer** | `answer_llm` | Generates response with `<USED_DOCS>` citation tags |

### History Support

| Component | History Usage |
|-----------|---------------|
| Router | Last 2 messages (100 chars each) |
| Reflection | Last 4 messages to resolve pronouns |
| Answer LLM | Last 4 messages as conversation context |
| ChitChat | Last 4 messages for natural flow |

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Điều-level chunking | Legal articles are semantic units |
| Hierarchy path in embeddings | Same article number differs across chapters |
| Q1 for reranking | Contextualized query captures intent better |
| 0.75 confidence threshold | High scores = clear match, skip LLM for speed |
| 4-message history limit | Balance context vs. token efficiency |

## 📋 Prerequisites

- **Python 3.10+**
- **Docker & Docker Compose**
- **Node.js 18+** (for frontend)
- **GPU (Optional)**: CUDA-compatible GPU for faster embedding

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repo-url>
cd law-chatbot

# Create virtual environments
cd law-crawler && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd ../langchain-backend && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Infrastructure Services

```bash
docker compose up -d
```

This starts:
- **Qdrant** (Vector DB) - `localhost:6333`
- **PostgreSQL** - `localhost:5432`
- **Prometheus** - `localhost:9090`
- **Grafana** - `localhost:3001`

### 3. Configure Environment

Edit `langchain-backend/.env`:

```env
# LLM Configuration
CHAT_MODEL=JunHowie/Qwen3-4B-GPTQ-Int4
URL=http://127.0.0.1:8000/v1
API_KEY=EMPTY

# External APIs
TAVILY_API_KEY=your_tavily_key
VOYAGE_API_KEY=your_voyage_key

# Database
DATABASE_URL=postgresql://root:123456789@localhost:5432/law_database

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Embedding Model
EMBEDDING_MODEL=AITeamVN/Vietnamese_Embedding
VECTOR_SIZE=1024
MAX_SEQ_LENGTH=2048

# Data Paths
PHAPDIEN_DIR=/path/to/law-chatbot/data/phap_dien
VBQPPL=/path/to/law-chatbot/data/vbqppl_content.json
QA_VBQPPL=/path/to/law-chatbot/data/qa_vbqppl_content.json
```

## 📚 Data Preparation

### Step 1: Prepare Pháp Điển Data

1. **Download** `BoPhapDienDienTu.zip` from the official source
2. **Unzip** to `law-crawler/BoPhapDienDienTu/`
3. **Run the crawler**:

```bash
cd law-crawler
python phapdien_crawler.py
```

**Output**: `data/phap_dien/Dieu.json`, `data/phap_dien/Chuong.json`

### Step 2: Crawl VBQPPL Documents

This fetches the legal documents referenced in Pháp Điển:

```bash
cd law-crawler
python vbqppl_crawler.py --input ../data/phap_dien/Dieu.json --output ../data/vbqppl_content.json
```

Options:
- `--phase 1` - Fetch HTML only
- `--phase 2` - Extract only (from cached HTML)
- `--limit N` - Process only N documents

### Step 3: Crawl QA Dataset References

For the evaluation dataset, crawl additional referenced documents:

```bash
cd law-crawler
python qa_dataset_crawler.py
```

**Output**: `data/qa_vbqppl_content.json`

## 💾 Data Ingestion

### Ingest into Qdrant (Vector Database)

```bash
cd langchain-backend
python ingest_qdrant.py
```

Creates collections:
- `phapdien_vietnamese_embedding` - Pháp Điển articles
- `vbqppl_vietnamese_embedding` - VBQPPL document sections

### Ingest into PostgreSQL (Full Documents)

```bash
cd langchain-backend
python ingest_psql.py

# To reset tables first:
python ingest_psql.py --drop
```

## 🖥️ Running the Backend

### Start vLLM Server (if using local model)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model JunHowie/Qwen3-4B-GPTQ-Int4 \
    --trust-remote-code
```

### Start FastAPI Backend

```bash
cd langchain-backend
python app.py
# Or with uvicorn:
uvicorn app:app --host 0.0.0.0 --port 8888 --reload
```

API available at `http://localhost:8888`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/chat` | POST | Chat with RAG |
| `/documents` | GET | List/Search documents |
| `/document/{id}` | GET | Get document by ID |

## 📊 Evaluation

### Dataset 1: du_lieu_luat_dataset.json

Generate system outputs:
```bash
cd langchain-backend
python eval.py --input ../data/du_lieu_luat_dataset.json --output ../data/evaluation_results.json
```

Compute metrics (retrieval & citation):
```bash
python compute_metrics.py
```

### Dataset 2: ALQAC-2025

1. **Ingest corpus**:
```bash
python ingest_alqac25.py
```

2. **Run evaluation**:
```bash
python eval_alqac25.py --input ../ALQAC-2025/alqac25_train.json --output ../data/alqac25_eval_results.json
```

3. **Compute metrics**:
```bash
python compute_alqac_metrics.py --results ../data/alqac25_eval_results.json
```

### Evaluation Metrics Explained

The system computes **Retrieval Metrics** and **Citation Metrics**:

#### Retrieval Metrics (RAG Pipeline Quality)

| Metric | Formula | Description |
|--------|---------|-------------|
| **Retrieval Recall** | `matched_gt / total_gt` | Percentage of ground truth citations found in retrieved context documents |
| **Retrieval Precision** | `relevant_docs / total_context_docs` | Percentage of retrieved documents that contain relevant citations |

#### Citation Metrics (LLM Answer Quality)

| Metric | Formula | Description |
|--------|---------|-------------|
| **Citation Recall** | `matched_gt / total_gt` | Percentage of ground truth citations the LLM actually cited in its answer |
| **Citation Precision** | `relevant_used / total_used` | Percentage of LLM's citations that are actually relevant |

#### Matching Methodology

The evaluation uses **fuzzy string matching** (SequenceMatcher ratio ≥ 0.5) because:
- Vietnamese legal citations have variations ("Điều 5" vs "Điều 5.")
- Hierarchy paths require reconstruction from document metadata
- Different formatting styles exist across documents

**Citation extraction** parses legal references like:
- "Điều 5 Luật Thanh niên 2020"
- "Nghị định 100/2019/NĐ-CP"
- "Khoản 2 Điều 3 Thông tư 01/2021/TT-BTP"

#### ALQAC-2025 Specific Metrics

| Metric | Description |
|--------|-------------|
| **Answer Accuracy** | Exact match for Đúng/Sai (True/False) and MCQ questions |
| **Retrieval F1** | `2 × P × R / (P + R)` for context documents |
| **Citation F1** | `2 × P × R / (P + R)` for used documents |

## 📁 Project Structure

```
law-chatbot/
├── docker-compose.yml      # Infrastructure services
├── prometheus.yml          # Prometheus config
├── data/
│   ├── phap_dien/         # Pháp Điển data
│   │   ├── Dieu.json      # Articles
│   │   └── Chuong.json    # Chapters
│   ├── vbqppl_content.json
│   ├── qa_vbqppl_content.json
│   └── du_lieu_luat_dataset.json
├── ALQAC-2025/            # Evaluation dataset
├── law-crawler/
│   ├── phapdien_crawler.py
│   ├── vbqppl_crawler.py
│   ├── qa_dataset_crawler.py
│   └── utils.py
├── langchain-backend/
│   ├── .env               # Configuration
│   ├── app.py             # FastAPI endpoints
│   ├── chat.py            # Chat chains
│   ├── rag.py             # RAG retrieval
│   ├── models.py          # DB models
│   ├── prompts.py         # System prompts
│   ├── utils.py           # Utility functions
│   ├── ingest_qdrant.py
│   ├── ingest_psql.py
│   ├── ingest_alqac25.py
│   ├── eval.py
│   ├── eval_alqac25.py
│   ├── compute_metrics.py
│   └── compute_alqac_metrics.py
└── frontend/              # React frontend
```

## � Monitoring (Prometheus & Grafana)

The project includes observability infrastructure for monitoring API performance.

### Architecture

```
┌──────────────────┐      scrape /metrics      ┌─────────────┐
│  FastAPI Backend │  ◄────────────────────────│  Prometheus │
│  (port 8888)     │         every 5s          │  (port 9090)│
└──────────────────┘                           └─────────────┘
                                                      │
                                                      │ data source
                                                      ▼
                                               ┌─────────────┐
                                               │   Grafana   │
                                               │  (port 3001)│
                                               └─────────────┘
```

### Prometheus

**Purpose**: Collects time-series metrics from the FastAPI backend.

**Configuration** (`prometheus.yml`):
```yaml
scrape_configs:
  - job_name: 'law-chatbot-backend'
    scrape_interval: 5s
    static_configs:
      - targets: ['host.docker.internal:8888']
```

**Available Metrics** (exposed at `http://localhost:8888/metrics`):
- `http_requests_total` - Total HTTP requests by method, path, status
- `http_request_duration_seconds` - Request latency histogram
- `http_requests_in_progress` - Current concurrent requests

**Access Prometheus UI**: http://localhost:9090

**Example Queries**:
```promql
# Request rate per second
rate(http_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))
```

### Grafana

**Purpose**: Visualize metrics with customizable dashboards.

**Access Grafana**: http://localhost:3001
- Default credentials: `admin` / `admin`

**Setup Dashboard**:
1. Go to **Connections** → **Data Sources** → **Add data source**
2. Select **Prometheus**
3. Set URL: `http://prometheus:9090`
4. Click **Save & Test**

**Recommended Dashboard Panels**:
| Panel | Query | Description |
|-------|-------|-------------|
| Request Rate | `rate(http_requests_total[1m])` | Requests per second |
| Latency P95 | `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))` | 95th percentile response time |
| Error Rate | `sum(rate(http_requests_total{status=~"5.."}[5m]))` | 5xx errors per second |
| Active Requests | `http_requests_in_progress` | Concurrent connections |

## �🔧 Troubleshooting

### Qdrant Connection Error
```bash
# Check if Qdrant is running
docker ps | grep qdrant
# Restart if needed
docker compose restart qdrant
```

### PostgreSQL Connection Error
```bash
# Check connection
docker exec -it law_postgres psql -U root -d law_database
```

### CUDA Out of Memory
- Reduce `batch_size` in ingestion scripts
- Use smaller embedding model

### Empty Chat Responses
- Verify vLLM server is running
- Check `URL` in `.env` matches vLLM endpoint

### Prometheus Not Collecting Metrics
```bash
# Check if backend exposes metrics
curl http://localhost:8888/metrics

# Check Prometheus targets
# Go to http://localhost:9090/targets
```
