# NIM Developer Support Agent

A retrieval-augmented, tool-calling assistant that helps developers integrate with **NVIDIA
NIM** (NVIDIA Inference Microservices). It answers questions grounded in NIM's real
documentation and validates a developer's config against actual deployment rules — rather than
producing a plausible-sounding but unchecked answer.

This mirrors the kind of workflow a developer-facing technical role does day to day: understand
a customer's setup, ground answers in real docs, and verify their configuration is actually
correct before telling them it's fine.

## What it does

- **Retrieves** relevant sections from NVIDIA's NIM documentation (deployment, model profiles,
  reasoning models, tool calling) using TF-IDF search over heading-based document chunks, so
  answers cite real, current information instead of relying on the model's training data.
- **Validates** developer-supplied configs with deterministic logic, not LLM guesswork:
  - `validate_docker_run` — checks a `docker run` command against NIM's actual required flags
    (`--gpus`, port mapping, API key, cache mount, user permissions).
  - `validate_chat_request` — checks a JSON request body against the correct shape for
    `/v1/chat/completions` vs. `/v1/completions` (the most common integration error in NIM's
    own docs).
- **Answers** through a NIM-hosted LLM call (OpenAI-compatible API), combining the retrieved
  documentation and the validator's structured output into one grounded response.

## How it works

```
docs/ (curated NIM documentation)
   │
   ▼
chunker.py     → splits docs into heading-based chunks (corpus.json)
   │
   ▼
retriever.py   → TF-IDF search: ranks chunks by relevance to a query
   │
   ▼
agent.py       → orchestrates the full loop:
   │              1. retrieve relevant doc chunks for the question
   │              2. call the LLM with those chunks + available tools
   │              3. if the model requests a tool, run tools.py's real
   │                 validator and feed the result back
   │              4. return a final answer grounded in both
   ▼
tools.py       → deterministic validators + their tool schemas
```

## Example interaction

```
you> Check this: docker run -it --rm -p 8000:8000 nvcr.io/nim/meta/llama-3.1-8b-instruct:latest

agent> That command is missing two required pieces:
       1. --gpus all — without it, the container can't access the GPU
       2. -e NGC_API_KEY=$NGC_API_KEY — required to pull the model on first launch
       You're also missing a cache volume mount (-v $LOCAL_NIM_CACHE:/opt/nim/.cache),
       so the container will re-download the model weights every time it restarts.
```

## Design notes

**Why TF-IDF instead of a neural embedding model for retrieval:** NIM's documentation is dense
with exact strings that matter character-for-character — `NIM_MODEL_PROFILE`,
`--enable-auto-tool-choice`, `/v1/chat/completions`. TF-IDF with a tokenizer that preserves
these tokens is a strong, low-dependency baseline for a small, technical corpus, and avoids
requiring a model-download step. Swapping in a dense embedding model later is a contained
change isolated to `retriever.py`.

**Why validate with code instead of asking the LLM to check it:** an LLM can miss a subtle
misconfiguration or state something is correct when it isn't. A deterministic function either
finds the missing flag or it doesn't. The model's role is to decide *when* to call the
validator and *how to explain* its result — the actual checking is real logic, not a guess.

## Project structure

```
nim-support-agent/
├── docs/                   # curated NIM documentation corpus
├── src/
│   ├── chunker.py          # splits docs into heading-based retrievable chunks
│   ├── retriever.py        # TF-IDF retrieval over the corpus
│   ├── tools.py             # deterministic validators + tool schemas
│   └── agent.py             # RAG + tool-calling orchestration
├── corpus.json              # generated chunk index
└── requirements.txt
```

## Try it

```bash
pip install -r requirements.txt
cd src
python agent.py --demo
```

Runs the full retrieval and validation pipeline without needing an API key. For a live LLM
response, set `NVIDIA_API_KEY` (free at [build.nvidia.com](https://build.nvidia.com)) and run
`python agent.py`.
