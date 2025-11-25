import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

subset = "arxiv_csAI_subset.json"
file = subset

def get_data():
  with open(file) as f:
    for line in f:
      yield line

data = get_data()

cols = ['id', 'authors', 'title', 'update_date', 'categories', 'abstract', 'doi']

interested_data = []
for line in data:
  paper = json.loads(line)
  interested_data.append({col: paper.get(col) for col in cols})

df = pd.DataFrame(interested_data)

df = df.drop('doi', axis=1)

df = df.dropna()

titles = list(df['title']
              .str.lower() # lowercase
              .str.replace('-', ' ', regex=False)
              .str.replace(r'[^\w\s]', ' ') # replace punction with a space
              .str.replace(r'\s+', ' ', regex=True) # remove whitespace
              .str.strip()
              )

abstracts = list(df['abstract']
              .str.lower() # lowercase
              .str.replace('-', ' ', regex=False)
              .str.replace(r'[^\w\s]', ' ') # replace punction with a space
              .str.replace(r'\s+', ' ', regex=True) # remove whitespace
              .str.strip()
              )

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def remove_stop_words(text):
  removed = [t for t in text.split() if t not in stop_words]
  return ' '.join(removed)

titles = [remove_stop_words(title) for title in titles]
abstracts = [remove_stop_words(abstract) for abstract in abstracts]

stemmer = SnowballStemmer('english')

def stem_text(text):
  stemmed = [stemmer.stem(word) for word in text.split()]
  return ' '.join(stemmed)

titles = [stem_text(title) for title in titles]
abstracts = [stem_text(abstract) for abstract in abstracts]

with open('prepared_titles.txt', 'w') as f:
  f.write('\n'.join(titles))
with open('prepared_abstracts.txt', 'w') as f:
  f.write('\n'.join(abstracts))