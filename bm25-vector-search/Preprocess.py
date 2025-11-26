import gzip
import json
import pickle
import pandas as pd
import re
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
