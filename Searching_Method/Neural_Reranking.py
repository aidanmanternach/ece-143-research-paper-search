import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def normalize_text(x):
    """
    Normalize various input types into a single text string.

    This function handles multiple possible formats for a document or text-like object
    (e.g., strings, dictionaries, list-like structures, custom objects with `page_content`).
    It attempts to extract meaningful textual content in a consistent way.

    Parameters
    ----------
    x : str
        The input query to normalize. 

    Returns
    -------
    str
        A normalized string representation of the input. Returns an empty string if `x` is None.
    """

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
    Compute a relevance score for a query-document pair using a Cross-Encoder model.

    The function tokenizes the query and document together, feeds them through a
    sequence classification model, and returns the scalar relevance score.

    Parameters
    ----------
    query : str
        The query text. Passed through `normalize_text`.
    doc : str
        The document text. Passed through `normalize_text`.
    tokenizer : PreTrainedTokenizer
        HuggingFace tokenizer used to encode the query-document pair.
    model : PreTrainedModel
        HuggingFace sequence classification model (Cross-Encoder).

    Returns
    -------
    float
        A scalar relevance score output by the model (logit).
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
    logits = outputs.logits.squeeze(-1)  # [1] -> scalar
    return float(logits.item())


def rerank_with_cross_encoder(query, candidates, tokenizer, model):
    """
    Rerank candidate documents using a Cross-Encoder relevance model.

    Each candidate is scored using `cross_encoder_score`, and the final output
    is sorted in descending order of relevance.

    Parameters
    ----------
    query : str
        The search query.
    candidates : list of (int, str)
        A list of `(doc_id, document_content)` pairs to rerank.
    tokenizer : PreTrainedTokenizer
        Tokenizer for encoding query-document pairs.
    model : PreTrainedModel
        Cross-Encoder sequence classification model.

    Returns
    -------
    list of (float, int)
        A list of `(score, doc_id)` tuples sorted by score descending.
    """

    scored = []
    for idx,doc in candidates:
        score = cross_encoder_score(query, doc, tokenizer, model)
        scored.append((score, idx))
    ranked = sorted(scored, key=lambda x: x[0], reverse=True)
    return ranked

def retrieve_and_rerank(
        query,
        candidates, 
        tokenizer_model =  "cross-encoder/ms-marco-MiniLM-L6-v2",
        cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L6-v2",
        final_k=10
    ):
    """
    Load a Cross-Encoder model and rerank candidate documents for a query.

    This function initializes the tokenizer and model from pretrained HuggingFace
    checkpoints, applies cross-encoder-based reranking, and returns the top results.

    Parameters
    ----------
    query : str or any
        The query to evaluate.
    candidates : list of (int, str)
        Candidate documents as `(doc_id, document_content)` tuples.
    tokenizer_model : str, optional
        HuggingFace model name or path for the tokenizer.
    cross_encoder_model : str, optional
        HuggingFace model name or path for the Cross-Encoder.
    final_k : int, optional
        Number of top-ranked results to return (default: 10).

    Returns
    -------
    list of (float, int)
        The top `final_k` ranked documents as `(score, doc_id)` pairs.
    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    cross_encoder = AutoModelForSequenceClassification.from_pretrained(cross_encoder_model)
    ranked_docs = rerank_with_cross_encoder(
        query,
        candidates,
        tokenizer,
        cross_encoder
    )

    return ranked_docs[:final_k]