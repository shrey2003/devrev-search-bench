# DevRev Search — Semantic Search over DevRev Knowledge Base

Semantic search system for the [DevRev Search](https://huggingface.co/datasets/devrev/search) dataset. Embeds ~65K knowledge base articles using OpenAI `text-embedding-3-small`, indexes them with FAISS, and retrieves relevant documents for test queries.

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/devrev-search.git
cd devrev-search
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. Run the Notebook

Open `devrev_search.ipynb` in Jupyter and run cells sequentially:

```bash
jupyter notebook devrev_search.ipynb
```

## Project Structure

```
devrev-search/
├── devrev_search.ipynb      # Main notebook: embed, index, search, evaluate
├── download_datasets.py     # Standalone script to download datasets as parquet
├── requirements.txt         # Python dependencies
├── test_queries_results.json # Search results for test queries
└── README.md
```

## What the Notebook Does

| Section | Description                                                                           |
| ------- | ------------------------------------------------------------------------------------- |
| **1–4** | Load & explore the 3 dataset splits (annotated queries, test queries, knowledge base) |
| **5**   | Generate embeddings with OpenAI `text-embedding-3-small` and build a FAISS index      |
| **6**   | Interactive search — query the knowledge base                                         |
| **7**   | Run evaluation on all test queries and save results in annotated-queries format       |
| **8**   | Load a previously saved index (skip re-embedding)                                     |

## Dataset

The [`devrev/search`](https://huggingface.co/datasets/devrev/search) dataset from Hugging Face contains:

- **`knowledge_base`** — ~65K article chunks from DevRev support docs
- **`annotated_queries`** — Queries paired with golden retrievals (train)
- **`test_queries`** — Held-out queries for evaluation

## Output Format

Results are saved in the same format as `annotated_queries`:

```json
{
  "query_id": "a97f93d2-...",
  "query": "end customer organization name not appearing...",
  "retrievals": [
    {
      "id": "ART-1234_KNOWLEDGE_NODE-5",
      "text": "...",
      "title": "..."
    }
  ]
}
```

## Cost Estimate

Embedding ~65K documents with `text-embedding-3-small` costs approximately **$0.50–$1.00** (at $0.02 per 1M tokens).

## License

MIT
