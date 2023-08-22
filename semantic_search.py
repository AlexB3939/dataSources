"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
from sentence_transformers import SentenceTransformer, util
import torch

# model_asym = 'msmarco-distilroberta-base-v3'
model_asym = 'msmarco-distilbert-base-v2'
model_sym = 'all-MiniLM-L6-v2'
embedder = SentenceTransformer(model_asym)


def get_sentence_embeddings(sentences):
    sentence_embeddings = embedder.encode(sentences, convert_to_tensor=True)
    return sentence_embeddings


def get_top_k_similar_sentences(query, corpus, k=5):
    # Find the closest k sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(k, len(corpus))
    corpus_embeddings = get_sentence_embeddings(corpus)
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        results.append((idx, score))
        # print(corpus[idx], "(Score: {:.4f})".format(score))
    return results
