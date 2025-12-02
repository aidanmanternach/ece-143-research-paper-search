from collections import defaultdict, Counter
import pickle

with open('prepared_titles.txt', 'r') as f:
  titles = f.read().split('\n')

with open('prepared_abstracts.txt', 'r') as f:
  abstracts = f.read().split('\n')

title_term_table = defaultdict(list)
abstract_term_table = defaultdict(list)

def add_title(text, doc_id):
  term_counts = Counter(text.split())
  for term, freq in term_counts.items():
    title_term_table[term].append((doc_id, freq))

def add_abstract(text, doc_id):
  term_counts = Counter(text.split())
  for term, freq in term_counts.items():
    abstract_term_table[term].append((doc_id, freq))

for i in range(len(titles)):
  add_title(titles[i], i)
  add_abstract(abstracts[i], i)

titles_file = 'titles_inverted_indices.pkl'
abstracts_file = 'abstracts_inverted_indices.pkl'

with open(titles_file, 'wb') as f:
  pickle.dump(title_term_table, f)

with open(abstracts_file, 'wb') as f:
  pickle.dump(abstract_term_table, f)

titles_lengths = {}
abstracts_lengths = {}

for i in range(len(titles)):
  titles_lengths[i] = len(titles[i].split())
  abstracts_lengths[i] = len(abstracts[i].split())

titles_lengths_file = 'titles_lengths.pkl'
abstracts_lengths_file = 'abstracts_lengths.pkl'

with open(titles_lengths_file, 'wb') as f:
  pickle.dump(titles_lengths, f)

with open(abstracts_lengths_file, 'wb') as f:
  pickle.dump(abstracts_lengths, f)