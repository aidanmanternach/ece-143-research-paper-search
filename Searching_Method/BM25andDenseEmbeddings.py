import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
import glob

def bm25(df, tf, dl, avg_dl, N):
    """
    Compute the BM25 score for a single term in a single document.

    Parameters
    ----------
    df : int
        Document frequency of the term (number of documents containing the term).
    tf : int or float
        Term frequency within the document.
    dl : int or float
        Length of the document.
    avg_dl : float
        Average document length across the corpus.
    N : int
        Total number of documents in the corpus.

    Returns
    -------
    float
        The BM25 relevance score for the term in the document.
    """

    k=1.2
    b=0.75
    return np.log((N/df) + 1) * (tf * (k + 1)) / (tf + k * (1 - b + (dl / avg_dl)))

def bm25_rankings(
        query,
        num_res,
        title_term_table,
        abstract_term_table,
        title_lengths,
        abstract_lengths,
        avg_doc_len_title,
        avg_doc_len_abs,
        N
    ):
    """
    Compute BM25-based rankings for documents given a query, using both title and abstract fields.

    Parameters
    ----------
    query : str
        The input search query (space-separated terms).
    num_res : int
        Number of top results to return.
    title_term_table : dict
        Mapping term -> list of (doc_id, term_frequency) for titles.
    abstract_term_table : dict
        Mapping term -> list of (doc_id, term_frequency) for abstracts.
    title_lengths : list or array
        Document lengths for title fields.
    abstract_lengths : list or array
        Document lengths for abstract fields.
    avg_doc_len_title : float
        Average title length across all documents.
    avg_doc_len_abs : float
        Average abstract length across all documents.
    N : int
        Total number of documents.

    Returns
    -------
    list of (int, float)
        Sorted list of `(doc_id, score)` pairs, highest score first. If fewer than
        `num_res` documents are scored, all results are returned.
    """

    title_weight = 0.5 # term appearing in title has more importance
    doc_scores = defaultdict(int)
    for term in query.split():
        title_term = title_term_table[term]
        for doc_id, tf in title_term:
            doc_scores[doc_id] += title_weight * bm25(len(title_term), tf, title_lengths[doc_id], avg_doc_len_title, N)
        abstract_term = abstract_term_table[term]
        for doc_id, tf in abstract_term:
            doc_scores[doc_id] += bm25(len(abstract_term), tf, abstract_lengths[doc_id], avg_doc_len_abs, N)

    ranked_docs = sorted(doc_scores.items(), key=lambda s: s[1], reverse=True)

    if len(ranked_docs) < num_res:
        return ranked_docs

    return ranked_docs[:num_res]

def combined_ranking(
        query, 
        num_res, 
        title_term_table,
        abstract_term_table,
        title_lengths,
        abstract_lengths,
        avg_doc_len_title,
        avg_doc_len_abs,
        N,
        model,
        document_embeddings,
        bm25_weight = 0.3,
        vector_weight=0.7
    ):
    """
    Combine BM25 scores and embedding-based semantic similarity to rank documents.

    Parameters
    ----------
    query : str
        Input search query.
    num_res : int
        Number of top results to return.
    title_term_table : dict
        Mapping term -> list of (doc_id, term_frequency) for titles.
    abstract_term_table : dict
        Mapping term -> list of (doc_id, term_frequency) for abstracts.
    title_lengths : list or array
        Document lengths for title fields.
    abstract_lengths : list or array
        Document lengths for abstract fields.
    avg_doc_len_title : float
        Average title length across documents.
    avg_doc_len_abs : float
        Average abstract length across documents.
    N : int
        Total number of documents.
    model : sentence-transformers model or similar
        Model used to encode the query into an embedding.
    document_embeddings : array-like, shape (num_docs, emb_dim)
        Precomputed embeddings for all documents.
    bm25_weight : float, optional
        Weight assigned to normalized BM25 scores (default: 0.3).
    vector_weight : float, optional
        Weight assigned to embedding similarity scores (default: 0.7).

    Returns
    -------
    list of (int, float)
        Top-ranked documents as `(doc_id, score)` pairs sorted by combined relevance.
    """
    
    bm25_res = bm25_rankings(
        query,num_res * 5,
        title_term_table,
        abstract_term_table,
        title_lengths,
        abstract_lengths,
        avg_doc_len_title,
        avg_doc_len_abs,
        N
    )

    bm25_scores = {doc_id: score for doc_id, score in bm25_res}

    # normalize bm25 to 0-1
    if bm25_scores:
        max_bm25 = max(bm25_scores.values())
        bm25_scores = {k: v/max_bm25 for k, v in bm25_scores.items()}

    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, document_embeddings)[0]

    combined_scores = {}
    for doc_id in range(len(title_lengths)):
        bm25_score = bm25_scores.get(doc_id, 0) * bm25_weight
        vector_score = similarities[doc_id] * vector_weight
        combined_scores[doc_id] = bm25_score + vector_score

    top_res = sorted(combined_scores.items(), key=lambda s: s[1], reverse=True)[:num_res]

    return top_res

def load_split_embeddings(path, pattern='document_embedding_part*.npz'):
    """
    Load and concatenate multiple embedding files saved in `.npz` parts.

    Parameters
    ----------
    path : str
        Directory containing the embedding part files.
    pattern : str, optional
        Glob pattern for selecting embedding files (default: 'document_embedding_part*.npz').

    Returns
    -------
    numpy.ndarray
        A single concatenated array of document embeddings.
    """

    files = sorted(glob.glob(path + pattern))
    
    chunks = []
    for file in files:
        data = np.load(file)
        chunks.append(data['embeddings'])
    
    return np.concatenate(chunks, axis=0)

def bm25_search(query, titles_bm25, abstracts_bm25, top_k=5):
    """
    Perform a BM25 search over titles and abstracts and return ranked document indices.

    This function queries two BM25 indexes—one for titles and one for abstracts—
    then combines their scores (with a higher weight on titles), ranks documents,
    and returns the indices of the top results.

    Parameters
    ----------
    query : str
        Search query string. It is lowercased and tokenized by simple space splitting.
    titles_bm25 : object
        BM25 index object for titles, expected to implement a `get_scores(query_tokens)` method.
    abstracts_bm25 : object
        BM25 index object for abstracts, expected to implement a `get_scores(query_tokens)` method.
    top_k : int or str, optional
        Number of top results to return. If `"all"`, all documents are returned.
        Default is 5.

    Returns
    -------
    list of int
        List of document indices ordered by relevance score (highest first).

    Notes
    -----
    The combined score gives titles a stronger influence in ranking.
    """

    query = query.lower().split(' ')

    titles_scores = titles_bm25.get_scores(query)
    abstract_scores = abstracts_bm25.get_scores(query)

    indexed_scores = [(t_s * 4 + a_s, idx) for idx, (t_s, a_s) in enumerate(zip(titles_scores, abstract_scores))]

    sorted_scores = sorted(indexed_scores, reverse=True)

    if top_k == 'all':
        papers_idx = [s[1] for s in sorted_scores]          
    else:
        papers_idx = [s[1] for s in sorted_scores[:top_k]]  

    return papers_idx


def clickable(df):
    """
    Convert rows of a DataFrame into clickable HTML-style arXiv links.

    Each row must contain at least:
    - 'id'  : arXiv identifier
    - 'title' : paper title
    - 'update_date' : last update date

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing search results with arXiv metadata.

    Returns
    -------
    list of str
        A list of HTML strings where each entry consists of:
        - a clickable link to the paper's PDF on arXiv
        - the paper title
        - the update date (displayed below the link)

    Example
    -------
    Returned HTML string format:
        '<a href="https://arxiv.org/pdf/XXXX.XXXXX.pdf">Title</a><br>2024-05-12'
    """

    search_res = []
    for _, row in df.iterrows():
        search_res.append(f'<a href="https://arxiv.org/pdf/{row["id"]}.pdf">{row["title"]}</a><br>{row["update_date"]}')

    return search_res