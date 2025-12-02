import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
import glob

def bm25(df, tf, dl, avg_dl, N):
    k=1.2
    b=0.75
    return np.log((N/df) + 1) * (tf * (k + 1)) / (tf + k * (1 - b + (dl / avg_dl)))

def bm25_rankings(query, num_res, title_term_table, abstract_term_table, title_lengths, abstract_lengths, avg_doc_len_title, avg_doc_len_abs, N):
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

def combined_ranking(query, num_res, title_term_table, abstract_term_table, title_lengths, abstract_lengths, avg_doc_len_title, avg_doc_len_abs, N, model, document_embeddings, bm25_weight = 0.3, vector_weight=0.7):
    bm25_res = bm25_rankings(query, num_res * 5, title_term_table, abstract_term_table, title_lengths, abstract_lengths, avg_doc_len_title, avg_doc_len_abs, N)

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
    files = sorted(glob.glob(path + pattern))
    
    chunks = []
    for file in files:
        data = np.load(file)
        chunks.append(data['embeddings'])
    
    return np.concatenate(chunks, axis=0)
def bm25_search(query, titles_bm25, abstracts_bm25,  top_k=5):
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
    search_res = []
    for _, row in df.iterrows():
        search_res.append(f'<a href="https://arxiv.org/pdf/{row['id']}.pdf">{row['title']}</a><br>{row['update_date']}')

    return search_res