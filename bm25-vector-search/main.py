import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import re
import os
import pickle
from collections import defaultdict, Counter
import sys
import glob
import gzip
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

def load_df(file):
    def get_data():
        if file.endswith('.gz'):
            with gzip.open(file, 'rt', encoding='utf-8') as f:
                for line in f:
                    yield line
        else:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    yield line

    data = get_data()

    cols = ['id', 'authors', 'title', 'update_date', 'categories', 'abstract']

    interested_data = []
    for line in data:
        paper = json.loads(line)
        interested_data.append({col: paper.get(col) for col in cols})

    df = pd.DataFrame(interested_data)
    df = df.dropna()

    return df

def load_pickle(file):
    if file.endswith('.gz'):
        with gzip.open(file, 'rb') as f:
            return pickle.load(f)
    else:
        with open(file, 'rb') as f:
            return pickle.load(f)
        
def load_pickle_files(path):
    titles_file = path + 'titles_inverted_indices.pkl'
    abstracts_file = path + 'abstracts_inverted_indices.pkl'

    with open(titles_file, 'rb') as f:
        title_term_table = pickle.load(f)

    with open(abstracts_file, 'rb') as f:
        abstract_term_table = pickle.load(f)

    titles_file = path + 'titles_lengths.pkl'
    abstracts_file = path + 'abstracts_lengths.pkl'

    with open(titles_file, 'rb') as f:
        title_lengths = pickle.load(f)

    with open(abstracts_file, 'rb') as f:
        abstract_lengths = pickle.load(f)

    return title_term_table, abstract_term_table, title_lengths, abstract_lengths

def prep_query(query, stop_words, stemmer):
    query = query.lower()  # lowercase
    query = query.replace('-', ' ')  # no regex parameter needed
    query = re.sub(r'[^\w\s]', ' ', query)  # use re.sub for regex
    query = re.sub(r'\s+', ' ', query)  # use re.sub for regex
    query = query.strip()

    removed = [t for t in query.split() if t not in stop_words]
    query = ' '.join(removed)

    stemmed = [stemmer.stem(word) for word in query.split()]
    query = ' '.join(stemmed)

    return query

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


if __name__ == '__main__':

    if not (len(sys.argv) > 2 and sys.argv[1] == 'search'):
        print('Incorrect cli arguments provided')
        print('Provide cli arguments in the form:')
        print('\tpython main.py search "{query}"')
        exit(1)

    if os.path.exists('./bm25-vector-search/data/'):
        path = './bm25-vector-search/data/'
    elif os.path.exists('./data/'):
        path = './data/'
    else:
        print('Error: Could not find data directory')
        print('Please run from either root or bm25-vector-search directory')
        exit(1)

    print('Loading DataFrame...')
    df = load_df(path + 'arxiv_csAI_subset.json.gz')

    print('Loading Pickle Files...')
    titles_file = path + 'titles_inverted_indices.pkl.gz'
    abstracts_file = path + 'abstracts_inverted_indices.pkl.gz'
    title_term_table = load_pickle(titles_file)
    abstract_term_table = load_pickle(abstracts_file)

    titles_file = path + 'titles_lengths.pkl.gz'
    abstracts_file = path + 'abstracts_lengths.pkl.gz'
    title_lengths = load_pickle(titles_file)
    abstract_lengths = load_pickle(abstracts_file)

    nltk.download('stopwords')

    stop_words = set(stopwords.words('english'))

    stemmer = SnowballStemmer('english')

    # BM25 Configuration Variables
    N = len(title_lengths)
    avg_doc_len_title = sum(title_lengths.values()) / N
    avg_doc_len_abs = sum(abstract_lengths.values()) / N

    # Load the split embedding files
    document_embeddings = load_split_embeddings(path)

    print('Loading Embedding Model...')
    model = SentenceTransformer('all-MiniLM-L6-v2')

    query = sys.argv[2]
    print(f'\nQuery: {query}')

    query = prep_query(query, stop_words, stemmer)
    combined_res = combined_ranking(query, 5, title_term_table, abstract_term_table, title_lengths, abstract_lengths, avg_doc_len_title, avg_doc_len_abs, N, model, document_embeddings)
    paper_idx = [s[0] for s in combined_res]
    res_df = df.iloc[paper_idx]
    i = 1
    for index, r in res_df.iterrows():
        print(f"{i}. {r['title']}")
        print(f"\thttps://arxiv.org/pdf/{r['id']}.pdf")
        print(f"\t{r['update_date']}\n")
        i += 1
