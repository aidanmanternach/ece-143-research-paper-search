import pandas as pd
import json

path = '.' # Configure to location arxiv data stored

original_dataset = path + "/arxiv-metadata-oai-snapshot.json"
with open(original_dataset, "r") as f:
  for i in range(3):
    line = f.readline()
    data = json.loads(line)
    print(data)

subset = "arxiv_csAI_subset.json"
target_category = "cs.AI"

total_lines = 0
kept_lines = 0

with open(original_dataset, "r") as infile, open(subset, "w") as outfile:
  for line in infile:
    total_lines += 1
    data = json.loads(line)
    cats = data.get('categories', '')
    cats_list = cats.split() if isinstance(cats, str) else []
    if target_category in cats_list:
      # write new jason
      json.dump(data, outfile)
      outfile.write("\n")
      kept_lines += 1

print(f"Kept {kept_lines} lines out of {total_lines} total lines")