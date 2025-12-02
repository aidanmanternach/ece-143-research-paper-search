import json
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ipywidgets as widgets
from IPython.display import display, clear_output
from wordcloud import WordCloud
import networkx as nx
from itertools import combinations

class TFIDF:
    """
    A class for analyzing a collection of research papers using TF-IDF vectorization,
    author network analysis, and community detection.

    Attributes
    ----------
    df : pd.DataFrame
        DataFrame containing paper metadata with columns 'id', 'title', 'abstract', 'authors', 'categories', 'update_date'.
    tfidf_vectorizer : TfidfVectorizer
        Scikit-learn TF-IDF vectorizer fitted on the combined title and abstract text.
    tfidf_matrix : scipy.sparse.csr_matrix
        TF-IDF matrix of shape (num_papers, num_features).
    """

    def __init__(self, df):
        """
        Initializes the TFIDF object by preparing text and author data and computing the TF-IDF matrix.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing paper information. Must include columns:
            'title', 'abstract', and 'authors'.
        """

        self.df = df
        self.df['combined_text'] = df['title'] + " " + df['abstract']

        def parse_authors(auth_str):
            if not isinstance(auth_str, str): return []
            return [a.strip() for a in auth_str.replace(' and ', ', ').split(',') if a.strip()]

        self.df['author_list'] = df['authors'].apply(parse_authors)

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['combined_text'])

    def search_papers(self, query, top_n=10):
        """
        Search for the most relevant papers given a query using cosine similarity on TF-IDF vectors.

        Parameters
        ----------
        query : str
            Text query to search for relevant papers.
        top_n : int, optional
            Number of top results to return (default is 10).

        Returns
        -------
        pd.DataFrame
            DataFrame of top N papers sorted by relevance with columns:
            ['id', 'title', 'authors', 'update_date', 'relevance_score', 'combined_text'].
        np.ndarray
            Array of cosine similarity scores for all papers with respect to the query.
        """

        query_vec = self.tfidf_vectorizer.transform([query])
        cosine_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = cosine_scores.argsort()[-top_n:][::-1]
        results = self.df.iloc[top_indices].copy()
        results['relevance_score'] = cosine_scores[top_indices]
        return results[['id', 'title', 'authors', 'update_date', 'relevance_score', 'combined_text']], cosine_scores

    def build_author_network(self, min_edge_weight=2, top_nodes_count=80):
        """
        Constructs an author collaboration network from the dataset.

        Nodes represent authors, edges represent co-authorships, and edge weights
        represent the number of shared papers.

        Parameters
        ----------
        min_edge_weight : int, optional
            Minimum number of collaborations required to keep an edge (default is 2).
        top_nodes_count : int, optional
            Number of top authors (by degree centrality) to retain in the subgraph (default is 80).

        Returns
        -------
        networkx.Graph
            Filtered subgraph containing top authors and edges above the weight threshold.
        """

        G = nx.Graph()
        for authors in self.df['author_list']:
            if len(authors) > 1:
                for u, v in combinations(sorted(authors), 2):
                    if G.has_edge(u, v):
                        G[u][v]['weight'] += 1
                    else:
                        G.add_edge(u, v, weight=1)

        edges_to_keep = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] >= min_edge_weight]
        G_filtered = G.edge_subgraph(edges_to_keep)
        degree_cent = nx.degree_centrality(G_filtered)
        top_authors = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:top_nodes_count]
        subgraph = G_filtered.subgraph([a[0] for a in top_authors])
        return subgraph

    def detect_communities(self, G_subgraph):
        """
        Detects communities within an author network using greedy modularity maximization.

        Parameters
        ----------
        G_subgraph : networkx.Graph
            Author network subgraph for community detection.

        Returns
        -------
        list of sets
            List of communities, where each community is represented as a set of author names.
        """

        from networkx.algorithms import community
        communities_detected = list(community.greedy_modularity_communities(G_subgraph))
        return communities_detected

    def generate_community_names(self, communities_detected):
        """
        Generates descriptive names for communities based on the most common paper categories
        associated with authors in each community.

        Parameters
        ----------
        communities_detected : list of sets
            Communities as returned by `detect_communities`.

        Returns
        -------
        dict
            Dictionary mapping community index to a descriptive name string.
        """

        author_to_categories = {}
        for idx, row in self.df.iterrows():
            authors = row['author_list']
            paper_categories = row['categories'].split()
            for author in authors:
                if author not in author_to_categories:
                    author_to_categories[author] = set()
                author_to_categories[author].update(paper_categories)

        community_names = {}
        for i, community_authors in enumerate(communities_detected):
            all_categories = []
            for author in community_authors:
                all_categories.extend(list(author_to_categories.get(author, set())))
            counts = collections.Counter(all_categories)
            if counts:
                most_common = counts.most_common(2)
                if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                    name = f"Primary Fields: {most_common[0][0]} & {most_common[1][0]}"
                else:
                    name = f"Primary Field: {most_common[0][0]}"
            else:
                name = "Diverse/Undefined Field"
            community_names[i] = name
        return community_names
    
    def search_doc_indices(self, query, top_n=10):
        """
        Searches for the top N documents matching a query and returns their indices and relevance scores.

        Parameters
        ----------
        query : str
            Text query to search for.
        top_n : int, optional
            Number of top results to return (default is 10).

        Returns
        -------
        list of tuples
            List of tuples [(doc_index, relevance_score), ...] sorted by relevance in descending order.
        """

        query_vec = self.tfidf_vectorizer.transform([query])
        cosine_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = cosine_scores.argsort()[-top_n:][::-1]
        return [(int(i), float(cosine_scores[i])) for i in top_indices]