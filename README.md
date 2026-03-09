# TaTaHuntsSignals (MVP)

![Tata](./assets/tata.png)

Portfolio-ready side project MVP built with **LangGraph + lightweight local RAG** to generate explainable single-ticker research reports.

## What It Does
- Accepts a US stock/ETF ticker input (e.g., `NVDA`, `AAPL`, `SPY`)
- Runs a modular LangGraph workflow
- Combines structured demo market/fundamental data + unstructured text evidence
- Computes deterministic 5-factor score (0-100)
- Generates scenario-based strategy notes with risks and confidence

## MVP Architecture
Workflow nodes:
1. Query Normalizer
2. Data Router
3. Structured Data Collector
4. Fundamentals Collector
5. Text Retrieval / RAG Node
6. Factor Engine
7. Score Composer
8. Report Generator
9. Guardrail / Critic Node

Core folders:
- `app/` UI + service entrypoint
- `agents/` graph node functions
- `graph/` LangGraph workflow wiring
- `data_providers/` interfaces + mock provider
- `rag/` local retrieval logic
- `scoring/` deterministic factor engine
- `models/` state schema
- `sample_data/` local demo data and text docs
- `tests/` baseline tests

## Quickstart
```bash
cd /home/nick/agentic-quant-copilot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app/main.py
```

Open the Streamlit URL shown in terminal.

## Demo Mode
- Default mode is `APP_MODE=demo`
- No API keys required
- Uses local JSON/text under `sample_data/`
- Unknown tickers fall back to `DEFAULT` snapshot + generic text

## Deterministic Score Framework
Each factor contributes **0-20**:
- Momentum / Trend
- Quality / Fundamentals
- Valuation
- Risk
- Sentiment / Catalyst

Total score = sum of factors (0-100).

## Testing
```bash
pytest -q
```

## Notes and Guardrails
- Research assistant only; not auto-trading or execution.
- Strategy output is scenario-oriented (bull/base/bear), not guaranteed prediction.
- Missing data lowers confidence and surfaces warnings.

## TODO (Next Iterations)
- Plug real data providers (Polygon, SEC filings APIs, news APIs)
- Replace keyword retrieval with embedding/vector index
- Add richer ETF-aware fundamental logic
- Add multi-ticker compare mode and watchlist ranking
- Add factor expression mini-engine extension points
- Export markdown/PDF report
- Improve critic node with stronger claim-evidence checks
