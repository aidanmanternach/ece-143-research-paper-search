import gzip
import json
import pickle
import pandas as pd
import re

def load_df(file):
    """
    Load a JSONL or gzipped JSONL file into a pandas DataFrame.

    The function reads line-by-line (supporting both `.jsonl` and `.jsonl.gz` formats),
    extracts a predefined subset of fields from each JSON object, and returns a DataFrame
    containing only those fields, with rows containing missing values removed.

    Parameters
    ----------
    file : str
        Path to the input file. Must be either:
        - a plain text `.jsonl` file
        - a gzip-compressed `.jsonl.gz` file

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the following columns:
        ['id', 'authors', 'title', 'update_date', 'categories', 'abstract'],
        with any rows missing these fields dropped.

    Notes
    -----
    * The file must contain one valid JSON object per line.
    * Compression is automatically handled based on the `.gz` extension.
    """


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
    """
    Load a Python object from a pickle file, supporting both regular and gzipped formats.

    Parameters
    ----------
    file : str
        Path to the pickle file. If it ends with `.gz`, gzip decompression is applied.

    Returns
    -------
    any
        The deserialized Python object stored in the pickle file.

    Raises
    ------
    pickle.UnpicklingError
        If the file is not a valid pickle.
    """

    if file.endswith('.gz'):
        with gzip.open(file, 'rb') as f:
            return pickle.load(f)
    else:
        with open(file, 'rb') as f:
            return pickle.load(f)
        
def load_pickle_files(path):
    """
    Load multiple BM25-related pickle files (inverted indices and document lengths)
    from a given directory.

    The function expects the following files to exist in `path`:
        - titles_inverted_indices.pkl
        - abstracts_inverted_indices.pkl
        - titles_lengths.pkl
        - abstracts_lengths.pkl

    Parameters
    ----------
    path : str
        Directory path containing the pickle files. Should end with a slash
        or otherwise form valid file paths when concatenated.

    Returns
    -------
    tuple
        (title_term_table, abstract_term_table, title_lengths, abstract_lengths), where:
        - title_term_table : dict
            Inverted index mapping terms to title term frequencies.
        - abstract_term_table : dict
            Inverted index mapping terms to abstract term frequencies.
        - title_lengths : list or dict
            Document lengths for titles.
        - abstract_lengths : list or dict
            Document lengths for abstracts.

    Raises
    ------
    FileNotFoundError
        If any of the required pickle files are missing.
    """


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
    """
    Preprocess a query string by normalizing, removing stop words, and applying stemming.

    The preprocessing pipeline includes:
        1. Lowercasing
        2. Replacing hyphens with spaces
        3. Removing punctuation
        4. Normalizing whitespace
        5. Removing stop words
        6. Applying the provided stemmer to each token

    Parameters
    ----------
    query : str
        The raw query string provided by the user.
    stop_words : set or list
        A collection of stop words to remove from the query.
    stemmer : object
        A stemmer implementing a `.stem(word)` method (e.g., NLTK's PorterStemmer).

    Returns
    -------
    str
        The cleaned and stemmed query string, ready for retrieval models.
    """

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
