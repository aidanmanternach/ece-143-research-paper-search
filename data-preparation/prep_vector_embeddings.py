from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

if torch.cuda.is_available():
  print(f"GPU available: {torch.cuda.get_device_name(0)}")
else:
  print("Using CPU")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

with open('prepared_titles.txt', 'r') as f:
  titles = f.read().split('\n')

with open('prepared_abstracts.txt', 'r') as f:
  abstracts = f.read().split('\n')

documents = []
for i in range(len(titles)):
  combined = f"{titles[i]} {abstracts[i]}"
  documents.append(combined)

print("Generating document embeddings...")
document_embeddings = model.encode(documents, show_progress_bar=True)
print(f"Generated embeddings shape: {document_embeddings.shape}")

embedding_file = 'document_embedding.npy'
np.save(embedding_file, document_embeddings)