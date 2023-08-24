import streamlit
from sentence_transformers import SentenceTransformer, util
import torch


class SentenceEmbedder:
    def __init__(self, model_name_or_path):
        self.model_name = model_name_or_path
        self.embedder = SentenceTransformer(model_name_or_path)
        self.cache = {}

    def get_sentence_embeddings(self, sentences: list[str]) -> torch.Tensor:
        embeddings = []
        for sentence in sentences:
            assert isinstance(sentence, str), f"Sentence must be a string, got {type(sentence)}"
            if sentence not in self.cache:
                self.cache[sentence] = self.embedder.encode([sentence])
            embeddings.append(self.cache[sentence])
        return torch.tensor(embeddings)

    @streamlit.cache_data(hash_funcs={"semantic_search.SentenceEmbedder": lambda sentence_embedder: hash(sentence_embedder.model_name)})
    def get_top_k_similar_sentence_indices(self, query: str, corpus: list[str], k=5) -> list[tuple[int, float]]:
        # Find the closest k sentences of the corpus for each query sentence based on cosine similarity
        top_k = min(k, len(corpus))

        corpus_embeddings = self.get_sentence_embeddings(corpus)
        corpus_embeddings = torch.squeeze(corpus_embeddings)
        # streamlit.write(f"corpus embeddings shape: {corpus_embeddings.size()}")
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        # streamlit.write(f"query embedding shape: {query_embedding.size()}")

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        results = [(idx_tensor.item(), score_tensor.item()) for score_tensor, idx_tensor in zip(top_results[0], top_results[1])]
        return results
