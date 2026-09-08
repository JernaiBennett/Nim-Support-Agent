"""
Retrieval layer: TF-IDF over the chunked NIM docs corpus.

Why TF-IDF instead of a neural embedding model: this environment has no access
to model-download endpoints (HuggingFace, etc.), and for a ~30-chunk technical
corpus full of exact API names, env vars, and flags (NIM_MODEL_PROFILE,
--enable-auto-tool-choice, /v1/chat/completions), sparse lexical matching is
actually a strong, dependency-light baseline. Swapping in a dense embedding
model later (e.g. sentence-transformers) is a drop-in change — see the
`embed_query` / `embed_corpus` seams below.
"""
import json
from pathlib import Path
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cuda_similarity import cosine_similarity_gpu, CUDA_AVAILABLE


@dataclass
class RetrievedChunk:
    id: str
    source: str
    heading: str
    text: str
    score: float


class NimDocRetriever:
    def __init__(self, corpus_path: Path, use_gpu: bool = True):
        raw = json.loads(corpus_path.read_text())
        self.chunks = raw
        self._texts = [f"{c['heading']} {c['text']}" for c in raw]

        # Custom tokenizer pattern keeps things like "/v1/chat/completions",
        # "NIM_MODEL_PROFILE", and "--enable-auto-tool-choice" intact instead
        # of shredding them into meaningless pieces.
        self.vectorizer = TfidfVectorizer(
            token_pattern=r"[A-Za-z0-9_\-/\.]+",
            stop_words="english",
            ngram_range=(1, 2),
        )
        self._matrix = self.vectorizer.fit_transform(self._texts)
        self._dense_matrix = self._matrix.toarray()

        # Use the CUDA kernel when a real GPU is present; on CPU-only
        # machines this stays False and every query uses the sklearn path
        # below, so behavior is identical either way — just the compute
        # backend differs.
        self.use_gpu = use_gpu and CUDA_AVAILABLE

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        query_vec = self.vectorizer.transform([query])

        if self.use_gpu:
            scores = cosine_similarity_gpu(query_vec.toarray()[0], self._dense_matrix)
        else:
            scores = cosine_similarity(query_vec, self._matrix)[0]

        ranked_idx = scores.argsort()[::-1][:top_k]

        results = []
        for i in ranked_idx:
            if scores[i] <= 0:
                continue
            c = self.chunks[i]
            results.append(RetrievedChunk(
                id=c["id"],
                source=c["source"],
                heading=c["heading"],
                text=c["text"],
                score=float(scores[i]),
            ))
        return results


if __name__ == "__main__":
    corpus_path = Path(__file__).parent.parent / "corpus.json"
    retriever = NimDocRetriever(corpus_path)

    test_queries = [
        "how do I enable tool calling with vllm",
        "what GPU memory do I need for a 70B model",
        "how to turn off chain of thought reasoning",
        "my chat completion request returns a 400 error",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        for r in retriever.retrieve(q, top_k=2):
            print(f"  [{r.score:.3f}] {r.source} :: {r.heading}")
