# Research Paper Search Engine

A three-stage IR (Information Retrieval) system for academic CS/AI papers.  
This system retrieves and ranks arXiv papers through three levels:

1. TF-IDF Keyword Retrieval
2. BM25 + Dense Embeddings (Hybrid Search)
3. Cross-Encoder Neural Re-ranking (State-of-the-Art)

Given a text query, the pipeline returns relevant papers ranked by semantic similarity and relevance.

---

## File Structure

```plaintext
├── main.py                       # Program entry - runs all 3 retrieval levels
├── Preprocess.py                 # Data loading + preprocessing functions
├── Searching_Method/
│   ├── TFIDF.py                  # TF-IDF vector search implementation
│   ├── BM25andDenseEmbeddings.py # BM25 + dense embedding ranking
│   ├── Neural_Reranking.py       # Cross-encoder reranking (Final stage)
├── data/                         # Processed metadata + embeddings
├── data-preparation/             # Scripts and steps used to filter raw dataset
├── visualization.ipynb           # Optional notebook for result analysis
├── requirements.txt              # Dependency list
└── .gitignore                    # Exclusion list for git tracking
```

## Dataset Description

The system uses the following dataset:

- [arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv/data)

From this dataset, only samples with category containing: `cs.ai` are extracted and processed into a `.json.gz` file before retrieval.

Relevant fields retained:

## How to Run

Run a query from terminal

python main.py search "attention mechanism"
Example console output:

```plaintext
Level 1: TF-IDF results
Level 2: BM25 + Embedding results
Level 3: Cross-Encoder reranked results
```

The top results from all three retrieval modules will be displayed.

## Third-Party Dependencies

External packages used in the system:

```plaintext
pandas
numpy
matplotlib
scikit-learn
nltk
sentence-transformers
transformers
torch
tqdm
gzip (built-in)
pickle (built-in)
json (built-in)
```

**Install via:**

```bash
pip install -r requirements.txt
```
