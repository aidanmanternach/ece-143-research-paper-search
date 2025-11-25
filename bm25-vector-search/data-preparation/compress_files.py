import gzip
import numpy as np

'''
The Non Compressed Files are too big to upload to Github
Must compress the files
'''

with open('arxiv_csAI_subset.json', 'r') as f_in:
    with gzip.open('arxiv_csAI_subset.json.gz', 'wt', encoding='utf-8') as f_out:
        f_out.write(f_in.read())

pickle_files = [
        'titles_inverted_indices.pkl',
        'abstracts_inverted_indices.pkl',
        'titles_lengths.pkl',
        'abstracts_lengths.pkl'
    ]

for pkl_file in pickle_files:
    input_file = pkl_file
    output_file = pkl_file + '.gz'
    with open(input_file, 'rb') as f_in:
        with gzip.open(output_file, 'wb') as f_out:
            f_out.write(f_in.read())

document_embeddings = np.load('document_embedding.npy')

num_chunks = 3
chunks = np.array_split(document_embeddings, num_chunks)

for i, chunk in enumerate(chunks):
    np.savez_compressed(f'document_embedding_part{i}.npz', embeddings=chunk)