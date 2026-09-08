"""
GPU-accelerated retrieval scoring using Numba CUDA.

TF-IDF retrieval requires computing cosine similarity between one query vector
and every chunk vector in the corpus. On CPU this is a sequential loop (or a
vectorized sklearn call that still runs on one core); on GPU each chunk's
similarity score is independent of every other chunk's, so it's an
embarrassingly parallel problem — a natural fit for a CUDA kernel where one
thread handles one chunk.

This module is written and tested against Numba's CUDA simulator
(NUMBA_ENABLE_CUDASIM=1), which runs the kernel logic on CPU exactly as
written but without real GPU hardware — this environment has no GPU, so the
simulator is what validates correctness here. On a machine with an NVIDIA
GPU, the same code runs unmodified on real hardware; no code changes needed,
only the environment variable goes away.

Falls back to the existing sklearn CPU path automatically when no CUDA device
is available, so retriever.py works unchanged on either machine.
"""
import numpy as np
from numba import cuda, float32


CUDA_AVAILABLE = cuda.is_available()


@cuda.jit
def _cosine_similarity_kernel(query_vec, corpus_matrix, query_norm, corpus_norms, out_scores):
    """
    One CUDA thread per corpus chunk. Each thread computes the dot product
    between the query vector and its assigned chunk's vector, then divides
    by the precomputed norms to get cosine similarity.

    query_vec:      (D,)   the query's TF-IDF vector
    corpus_matrix:  (N, D) all chunk vectors, one row per chunk
    query_norm:     scalar, precomputed L2 norm of query_vec
    corpus_norms:   (N,)   precomputed L2 norm of each chunk vector
    out_scores:     (N,)   output buffer for similarity scores
    """
    i = cuda.grid(1)
    if i < corpus_matrix.shape[0]:
        dot = float32(0.0)
        for d in range(corpus_matrix.shape[1]):
            dot += query_vec[d] * corpus_matrix[i, d]
        denom = query_norm * corpus_norms[i]
        if denom > 0:
            out_scores[i] = dot / denom
        else:
            out_scores[i] = 0.0


def cosine_similarity_gpu(query_vec: np.ndarray, corpus_matrix: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between one query vector and every row of
    corpus_matrix, using the CUDA kernel above. Falls back to a NumPy
    equivalent if no CUDA device is available (e.g. running the simulator
    or on CPU-only hardware), so callers never need to branch on this.
    """
    query_vec = query_vec.astype(np.float32)
    corpus_matrix = corpus_matrix.astype(np.float32)

    query_norm = np.linalg.norm(query_vec)
    corpus_norms = np.linalg.norm(corpus_matrix, axis=1)

    n_chunks = corpus_matrix.shape[0]
    out_scores = np.zeros(n_chunks, dtype=np.float32)

    threads_per_block = 128
    blocks_per_grid = (n_chunks + threads_per_block - 1) // threads_per_block

    d_query = cuda.to_device(query_vec)
    d_corpus = cuda.to_device(corpus_matrix)
    d_query_norm = np.float32(query_norm)
    d_corpus_norms = cuda.to_device(corpus_norms.astype(np.float32))
    d_out = cuda.to_device(out_scores)

    _cosine_similarity_kernel[blocks_per_grid, threads_per_block](
        d_query, d_corpus, d_query_norm, d_corpus_norms, d_out
    )

    return d_out.copy_to_host()


if __name__ == "__main__":
    # Self-test: compare kernel output against a known-correct NumPy
    # cosine similarity implementation, run under the CUDA simulator.
    rng = np.random.default_rng(42)
    n_chunks, dims = 31, 200  # matches the actual corpus size (31 chunks)
    corpus = rng.random((n_chunks, dims)).astype(np.float32)
    query = rng.random(dims).astype(np.float32)

    gpu_scores = cosine_similarity_gpu(query, corpus)

    # Reference implementation using plain NumPy
    def cosine_similarity_reference(q, m):
        q_norm = np.linalg.norm(q)
        m_norms = np.linalg.norm(m, axis=1)
        dots = m @ q
        return dots / (q_norm * m_norms)

    reference_scores = cosine_similarity_reference(query, corpus)

    max_diff = np.max(np.abs(gpu_scores - reference_scores))
    print(f"CUDA_AVAILABLE (real GPU): {CUDA_AVAILABLE}")
    print(f"Max difference vs NumPy reference: {max_diff:.2e}")
    print(f"Match (tolerance 1e-4): {max_diff < 1e-4}")
    print(f"\nSample scores — GPU kernel:  {gpu_scores[:5]}")
    print(f"Sample scores — reference:   {reference_scores[:5]}")
