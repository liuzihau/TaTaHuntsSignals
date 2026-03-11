# TaTaHuntsSignals

**A news-driven stock momentum ranking system for US top 3000 stocks.**

Combines financial news NLP (FinBERT) + machine learning (LightGBM) to predict short-term stock momentum (velocity of price movement, not just direction).

---

## 🎯 Project Goal

Predict which stocks will move fastest in the next 30/90/180 days using:
1. News sentiment + event signals (from financial articles)
2. News flow aggregation (past N days per stock)
3. Lightweight ML model trained on historical momentum

**Output:** Ranked list of 3000 US stocks by momentum score.

---

## 🏗️ Architecture

```
News Articles (1000+)
    ↓ FinBERT Encoder (financial-tuned BERT)
768-dim embeddings
    ↓ Aggregate per stock (7/14/30/60 day windows)
News feature vectors
    ↓ LightGBM Model (2 outputs)
├─ predict: next 30/90/180 days return %
└─ predict: next 30/90/180 days return % / day (momentum velocity)
    ↓
Rank 3000 stocks by momentum score
```

---

## ⚡ Setup & Environment

### Prerequisites
- **PostgreSQL** running locally (or Docker)
- **Conda** environment manager
- **Python 3.10+**

### 1. Activate Conda Environment
```bash
# Use the existing tata-hunts-signals environment
conda activate tata-hunts-signals

# Or create new (one-time):
# conda create -n tata-hunts-signals python=3.11
# conda activate tata-hunts-signals
```

### 2. Install Dependencies
```bash
cd ~/TaTaHuntsSignals
pip install -r requirements.txt
```

### 3. Initialize PostgreSQL Database
```bash
# Create database and user (one-time)
psql -U postgres << 'EOF'
CREATE DATABASE tatahunts_rag ENCODING 'UTF8';
CREATE USER tatahunts_user WITH PASSWORD 'tatahunts_password';
GRANT ALL PRIVILEGES ON DATABASE tatahunts_rag TO tatahunts_user;
EOF

# Create schema and tables
psql -U tatahunts_user -d tatahunts_rag -f storage/postgres_schema.sql

# Verify connection
psql -U tatahunts_user -d tatahunts_rag -c "SELECT 1"
# Expected output: 1
```

---

## 🚀 Quick Start: News Crawling + FinBERT Encoding

### Step 1: Crawl Recent News (1-2 months)
```bash
conda activate tata-hunts-signals
cd ~/TaTaHuntsSignals

# Crawl top 50 stocks (RSS + article body extraction)
python3 crawl_recent_news.py --workers 5 --sleep-ms 500

# Options:
# --tickers "AAPL,MSFT,NVDA"  # Custom ticker list
# --skip-detail               # RSS only, no body extraction
# --workers 10                # Parallel detail crawlers (default: 5)
# --sleep-ms 1000             # Delay between RSS crawls

# Expected output: 500-2000 articles in DB (depends on feed activity)
```

### Step 2: Encode Articles with FinBERT (768-dim)
```bash
python3 run_finbert.py --batch-size 32 --device cuda

# Options:
# --batch-size 64             # Larger batch = faster (if GPU memory available)
# --device cpu                # Use CPU instead of CUDA
# --limit 100                 # Test mode: encode only 100 articles
# --force-reencode            # Re-encode articles that already have embeddings

# Expected output: 1000+ articles with 768-dim embeddings stored in PostgreSQL
```

### Step 3: Verify Results
```bash
python3 << 'EOF'
from storage.postgres_client import PostgreSQLClient

pg = PostgreSQLClient()
with pg._require_conn().cursor() as cur:
    cur.execute("SELECT COUNT(*) FROM articles WHERE has_embedding = TRUE")
    embedded = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM articles")
    total = cur.fetchone()[0]

print(f"✅ {embedded}/{total} articles embedded with FinBERT")
pg.close()
EOF
```

---

## 📊 Crawling Examples

### Example A: Full Crawl (Recommended for first run)
```bash
# Crawl 30 major tech stocks, with full article body extraction
python3 crawl_recent_news.py \
    --tickers "AAPL,MSFT,NVDA,GOOGL,AMZN,META,TSLA" \
    --workers 5 \
    --sleep-ms 500
```

### Example B: Fast Crawl (RSS only, no body)
```bash
# Quick RSS feed crawl, skip detailed body extraction
python3 crawl_recent_news.py \
    --skip-detail \
    --sleep-ms 200
```

### Example C: Custom Stock List
```bash
# Crawl specific sectors
python3 crawl_recent_news.py \
    --tickers "XOM,CVX,COP,MPC,PSX,EQNR" \
    --workers 8
```

---

## ✅ Current Status

| Phase | Task | Status | Result |
|-------|------|--------|--------|
| 1-2 | News Crawling | ✅ DONE | RSS crawler + detail crawler |
| 4-5 | FinBERT Encoding | ✅ DONE | 768-dim vectors in pgvector |
| 6 | Feature Aggregation | ⏳ NEXT | Aggregate news → stock features |
| 7 | Model Training | 📋 PLANNED | LightGBM training |
| 8 | Ranking Engine | 📋 PLANNED | Real-time 3000-stock ranking |

See `PROJECT_STATUS.md` for detailed roadmap.

---

## 📁 Folder Structure

```
TaTaHuntsSignals/
├── data_sources/              # Crawlers
│   ├── yahoo_rss_crawler.py   # RSS feed crawler
│   └── news_detail_crawler.py # Article body extractor
├── storage/
│   ├── postgres_client.py     # DB CRUD operations
│   └── postgres_schema.sql    # Schema (pgvector support)
├── embedding/
│   └── finbert_encoder.py     # 768-dim FinBERT encoder ✅
├── aggregation/
│   └── news_aggregator.py     # [Phase 6] News aggregation
├── training/
│   ├── data_prep.py           # [Phase 7] Training data prep
│   ├── momentum_model.py       # [Phase 7] LightGBM model
│   └── evaluator.py           # [Phase 7] Model evaluation
├── inference/
│   └── ranking_engine.py       # [Phase 8] 3000-stock ranking
├── crawl_recent_news.py        # Crawling driver script
├── run_finbert.py              # FinBERT encoding driver
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── PROJECT_STATUS.md           # Full roadmap & architecture
```

---

## 🔧 Tech Stack

- **Encoder:** FinBERT (financial-domain BERT, 768-dim)
- **Storage:** PostgreSQL + pgvector (vector extension)
- **Model:** LightGBM (gradient boosting)
- **Framework:** Python 3.10+, PyTorch, Transformers
- **Environment:** Conda (recommended)

---

## 🧪 Testing

```bash
# Test FinBERT encoder (8 unit tests)
python3 test_finbert.py
# Expected: 8/8 PASSED ✅

# Run all tests
pytest tests/ -v
```

---

## ⏱️ Expected Runtime

| Step | Time | Notes |
|------|------|-------|
| RSS Crawl | 2-5 min | 50 tickers × multiple feeds |
| Detail Crawl | 10-30 min | Parallel (5 workers), network dependent |
| FinBERT Encoding | 5-15 min | 1000 articles, batch=32, GPU enabled |
| **Total** | **20-50 min** | End-to-end crawl + encoding |

---

## 🎓 Contributing

1. Read `PROJECT_STATUS.md` for roadmap
2. Activate conda: `conda activate tata-hunts-signals`
3. Implement feature (Phase 6-8)
4. Test: `pytest tests/`
5. Commit with clear message

---

## 📞 Documentation

- **PROJECT_STATUS.md** — Full roadmap, architecture, technical decisions
- **postgres_schema.sql** — Database schema (pgvector, 3-layer design)
- **run_finbert.py** — FinBERT encoding examples
- **crawl_recent_news.py** — Crawling driver with options

---

## 🚨 Troubleshooting

### PostgreSQL Connection Failed
```bash
# Check if PostgreSQL is running
psql -U tatahunts_user -d tatahunts_rag -c "SELECT 1"

# If not running (macOS):
brew services start postgresql

# Or use Docker:
docker run -d --name tatahunts-pg \
  -e POSTGRES_PASSWORD=tatahunts_password \
  -p 5432:5432 postgres:15
```

### CUDA Not Available
FinBERT encoder auto-detects GPU; falls back to CPU if unavailable.
```bash
# Force CPU mode (slower but always works)
python3 run_finbert.py --device cpu --batch-size 8
```

### psycopg2 Install Issues
```bash
# Install system dependencies first
# macOS:
brew install postgresql

# Ubuntu:
sudo apt-get install libpq-dev python3-dev

# Then:
pip install psycopg2-binary
```

---

**Status:** Phase 1-2 crawling in progress 🔄 | Phase 4-5 ready to encode ⏳ | Phase 6-8 planned 📋 | ETA: 2-3 weeks after encoding
