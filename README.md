# Open GR--WM

Local-first Graph RAG workspace for ingesting PDFs and web URLs, building a knowledge graph, and chatting with grounded answers using Ollama models.

## What It Does
- Ingests PDF and URL sources locally.
- Builds a fact-dense graph with text + selective vision extraction.
- Persists graph/index artifacts for reload/unload/delete/export/import.
- Answers questions with strict grounding and citation checks.

## Local Stack
- UI: Streamlit
- Graph: NetworkX + PyVis
- Embeddings: `all-MiniLM-L6-v2` (Sentence Transformers)
- Local models via Ollama:
  - Scanner: `llama3.2:latest`
  - Vision: `llama3.2-vision:latest`
  - Brain: `deepseek-r1:14b`

## Prerequisites
1. Python 3.10+
2. Ollama running locally
3. Models pulled:
   - `ollama pull llama3.2:latest`
   - `ollama pull llama3.2-vision:latest`
   - `ollama pull deepseek-r1:14b`

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## Repository Structure
- `app.py`: Streamlit app orchestration and UI
- `graphrag_v2/`: chunking, extraction, canonicalization, indexing, retrieval, evals, storage
- `docs/`: PRD, TRD, system design, and feature slide docs

## Documentation
- Docs index: [`docs/README.md`](docs/README.md)
- PRD: [`docs/prd/Open_GR_WM_PRD.md`](docs/prd/Open_GR_WM_PRD.md)
- TRD: [`docs/trd/Open_GR_WM_System_Design_TRD.md`](docs/trd/Open_GR_WM_System_Design_TRD.md)
- System Design PDF: [`docs/system/Open_GR_WM_System_Design.pdf`](docs/system/Open_GR_WM_System_Design.pdf)
- Feature Slides PDF: [`docs/slides/Open_GR_WM_Product_Features_Slides.pdf`](docs/slides/Open_GR_WM_Product_Features_Slides.pdf)

## Notes
- Runtime local artifacts are intentionally not versioned (`rag_store/`, caches, temporary files).
- This repo is designed for fully local execution.
