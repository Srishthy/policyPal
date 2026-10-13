# PolicyPal

An AI assistant for Indian health insurance. Upload your policy PDF and ask what it covers, or ask any general health insurance question. Answers cite the exact clause and page they come from.

## Features

- Answers from your own policy, with clause and page citations
- General questions answered from IRDAI regulations, with web search as a fallback
- Waiting-period and claim-payout calculators (no LLM maths)
- Aadhaar, PAN, phone and email hidden before anything is sent to the LLM

## Tech stack

LangGraph · LangChain · Groq · FAISS + BM25 (hybrid search) · BGE embeddings · Tavily · Streamlit

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # add your GROQ_API_KEY and TAVILY_API_KEY
```

Put the IRDAI PDFs in `data/irdai/` and build the index once:

```bash
python -m scripts.ingest_rules
```

## Run

```bash
streamlit run streamlit_app.py
```

## Tests

Run without API keys:

```bash
python tests/make_demo_policy.py
python tests/test_offline.py
```

## Disclaimer

PolicyPal gives general information, not legal or financial advice. Always confirm with your insurer.