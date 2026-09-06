# Getting Started with NIM: A Developer Support Agent

A small retrieval-augmented, tool-calling assistant that helps developers integrate with
**NVIDIA NIM** (NVIDIA Inference Microservices) — the same kind of workflow a Solutions
Architect walks a customer through: answer questions grounded in the real docs, and actually
validate their config instead of guessing whether it's right.

## What it does

1. **Retrieves** relevant sections from a curated set of real NVIDIA NIM documentation
   (getting started, model profiles, reasoning models, tool calling) using TF-IDF search over
   heading-based chunks.
2. **Validates** developer-supplied artifacts with real logic, not vibes:
   - `validate_docker_run` — checks a `docker run` command for the flags NIM actually requires
     (`--gpus`, `-p`, an API key, cache mount, user permissions).
   - `validate_chat_request` — checks a JSON request body against `/v1/chat/completions` vs
     `/v1/completions` shape rules (the #1 error in NIM's own docs).
3. **Answers** through an LLM call to a NIM-hosted model (OpenAI-compatible API), grounded in
   both the retrieved docs and the validator's structured output.

## Project layout

```
nim-support-agent/
├── docs/                   # curated NIM documentation corpus (real doc content)
├── src/
│   ├── chunker.py          # splits docs into heading-based retrievable chunks
│   ├── retriever.py        # TF-IDF retrieval over the corpus
│   ├── tools.py            # deterministic validators + tool schemas
│   └── agent.py            # RAG + tool-calling orchestration
├── corpus.json             # generated chunk index (run chunker.py to rebuild)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
cd src
python chunker.py      # builds corpus.json from docs/
```

## Try it without an API key

```bash
python agent.py --demo
```

This runs the retrieval and validation steps directly so you can see the pipeline work end to
end, without needing NVIDIA credentials.

## Run it for real

1. Get a free API key at [build.nvidia.com](https://build.nvidia.com/) (NVIDIA API Catalog).
2. Export it:
   ```bash
   export NVIDIA_API_KEY=nvapi-xxxxxxxx
   ```
3. Run the interactive agent:
   ```bash
   python agent.py
   ```
4. Ask it things like:
   - "Why is my docker run command not starting the NIM server?"
   - "How do I turn off chain-of-thought output for a Nemotron model?"
   - "Does tool calling work with the multi-LLM container?"
   - Paste a `docker run ...` command or a JSON request body and ask it to check it.

To point at a self-hosted NIM container instead of NVIDIA's hosted API catalog:
```bash
python agent.py --base-url http://localhost:8000/v1 --model meta/llama-3.1-8b-instruct
```

## Why TF-IDF instead of embeddings

This is a deliberate choice, not a shortcut taken for lack of time: NIM's docs are dense with
exact strings that matter character-for-character — `NIM_MODEL_PROFILE`,
`--enable-auto-tool-choice`, `/v1/chat/completions`. TF-IDF with a tokenizer that preserves
these tokens is a strong, dependency-light baseline for a small technical corpus like this one.
Swapping in a dense embedding model is a contained change — see the seams in `retriever.py`.

## Extending this

- Add more doc pages to `docs/` and re-run `chunker.py` to grow the corpus.
- Add more validators to `tools.py` (e.g. a profile-ID checker, a Kubernetes YAML linter).
- Swap `retriever.py`'s TF-IDF for a real embedding model once network access to a model host
  is available.
