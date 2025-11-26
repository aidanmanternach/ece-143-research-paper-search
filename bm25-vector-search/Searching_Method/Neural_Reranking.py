import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def normalize_text(x):
  if x is None:
      return ""

  if isinstance(x, str):
      return x

  if hasattr(x, "page_content"):
      return x.page_content

  if isinstance(x, dict):
      for key in ["abstract", "text", "content", "body"]:
          if key in x and isinstance(x[key], str):
              return x[key]
      return " ".join(str(v) for v in x.values())

  if isinstance(x, (list, tuple, set)):
      return " ".join(str(e) for e in x)

  return str(x)

def cross_encoder_score(query, doc, tokenizer, model):
  """
  Return a embedding similarilty score between a query and a document.
  """
  query = normalize_text(query)
  doc = normalize_text(doc)
  encoded = tokenizer(
        query,
        doc,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )

  with torch.no_grad():
    outputs = model(**encoded)
  cls_emb = outputs.last_hidden_state[:, 0, :] # feature of the sentence
  cls_emb = F.normalize(cls_emb, dim=1)
  return cls_emb.squeeze(0)  # cross-encoding use CLS as score

def rerank_with_cross_encoder(query, candidates, tokenizer, model):
  """
  Rerank the candidate documents with a cross-encoder.
  """
  scored = []
  for doc in candidates:
    score = cross_encoder_score(query, doc, tokenizer, model)
    scored.append((score, doc))
  ranked = sorted(scored, key=lambda x: torch.dot(x[0], x[0]), reverse=True)

  return [doc for _, doc in ranked]

def retrieve_and_rerank(query, candidates,  tokenizer_model =  "bert-base-uncased", cross_encoder_model = "bert-base-uncased", top_k=100, final_k=10):
  """
  Rerank using cross-encoder and return final_k best docs.
  """
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
  cross_encoder = AutoModel.from_pretrained(cross_encoder_model)  
  ranked_docs = rerank_with_cross_encoder(
        query,
        candidates,
        tokenizer,
        cross_encoder
    )

  return ranked_docs[:final_k]