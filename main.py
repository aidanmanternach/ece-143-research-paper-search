import os, sys, nltk, time
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from Preprocess import load_df, load_pickle, load_pickle_files,prep_query
from Searching_Method.Neural_Reranking import retrieve_and_rerank
from Searching_Method.BM25andDenseEmbeddings import load_split_embeddings, combined_ranking
from Searching_Method.TFIDF import TFIDF
if __name__ == '__main__':
    # preprocess
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

    print('Loading Embedding Model...')
    model = SentenceTransformer('all-MiniLM-L6-v2')

    raw_query = sys.argv[2]
    print(f'\nQuery: {raw_query}')

    query = prep_query(raw_query, stop_words, stemmer)


    # level 1 : TF-IDF
    
    print('Level 1 : TF-IDF model ')
    tfidf_model = TFIDF(df)
    t1_start = time.time()
    tfidf_results_df, _ = tfidf_model.search_papers(raw_query, top_n=5)
    t1_end = time.time()
    for i, (_, row) in enumerate(tfidf_results_df.iterrows(), start=1):
        print(f"{i}. {row['title']}")
        print(f"\thttps://arxiv.org/pdf/{row['id']}.pdf")
        print(f"\t{row['update_date']} (TF-IDF score: {row['relevance_score']:.4f})\n")
    
    print(f"Level 1 elapsed time: {t1_end - t1_start:.3f} seconds\n")
    # level 2 : BM25 Configuration Variables
    print('Level 2 : BM25 with TF-IDF model ')
    N = len(title_lengths)
    avg_doc_len_title = sum(title_lengths.values()) / N
    avg_doc_len_abs = sum(abstract_lengths.values()) / N

    # Load the split embedding files
    document_embeddings = load_split_embeddings(path)

    t2_start = time.time()
    combined_res = combined_ranking(query, 5, title_term_table, abstract_term_table, title_lengths, abstract_lengths, avg_doc_len_title, avg_doc_len_abs, N, model, document_embeddings)
    t2_end = time.time()
    paper_idx = [s[0] for s in combined_res]
    res_df = df.iloc[paper_idx]
    i = 1
    for index, r in res_df.iterrows():
        print(f"{i}. {r['title']}")
        print(f"\thttps://arxiv.org/pdf/{r['id']}.pdf")
        print(f"\t{r['update_date']}\n")
        i += 1

    print(f"Level 2 elapsed time: {t2_end - t2_start:.3f} seconds\n")
    #  level 3 : Neural Reranking with Cross-Encoders
    print("Level 3 : Neural Reranking with Cross-Encoders")
    t3_start = time.time()
    candidates = combined_ranking(query, 100, title_term_table, abstract_term_table, title_lengths, abstract_lengths, avg_doc_len_title, avg_doc_len_abs, N, model, document_embeddings)
   
    candidates_with_text = []
    # put in the text to rerank 
    for doc_id, score in candidates:
        abstract = df.iloc[doc_id].get("abstract", "")
        title    = df.iloc[doc_id].get("title", "")
        text     = title + " " + abstract            
        candidates_with_text.append((doc_id, text))

    reranked = retrieve_and_rerank(
        query,
        candidates_with_text,
        tokenizer_model = "bert-base-uncased",
        cross_encoder_model = "bert-base-uncased"
    )
    t3_end = time.time()
    paper_idx = [s[1] for s in reranked[:5]]

    res_df = df.iloc[paper_idx]

    i = 1
    for index, r in res_df.iterrows():
        print(f"{i}. {r['title']}")
        print(f"\thttps://arxiv.org/pdf/{r['id']}.pdf")
        print(f"\t{r['update_date']}\n")
        i += 1
    print(f"Level 3 elapsed time: {t3_end - t3_start:.3f} seconds\n")