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
    def __init__(self, df):
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
        query_vec = self.tfidf_vectorizer.transform([query])
        cosine_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = cosine_scores.argsort()[-top_n:][::-1]
        results = self.df.iloc[top_indices].copy()
        results['relevance_score'] = cosine_scores[top_indices]
        return results[['id', 'title', 'authors', 'update_date', 'relevance_score', 'combined_text']], cosine_scores

    def build_author_network(self, min_edge_weight=2, top_nodes_count=80):
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
        from networkx.algorithms import community
        communities_detected = list(community.greedy_modularity_communities(G_subgraph))
        return communities_detected

    def generate_community_names(self, communities_detected):
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
        return in same type [(doc_index, relevance_score), ...]
        """
        query_vec = self.tfidf_vectorizer.transform([query])
        cosine_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = cosine_scores.argsort()[-top_n:][::-1]
        return [(int(i), float(cosine_scores[i])) for i in top_indices]