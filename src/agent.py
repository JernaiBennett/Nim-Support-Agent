"""
NIM Developer Support Agent.

Combines:
  1. RAG: retrieves relevant chunks from the NVIDIA NIM docs corpus (TF-IDF)
  2. Tool calling: lets the model validate a docker run command or a chat
     request body against real NIM rules, rather than eyeballing it
  3. Chat: calls a NIM-hosted LLM (OpenAI-compatible endpoint) to produce the
     final answer, grounded in the retrieved doc chunks and any tool results

Run modes:
  - `python src/agent.py --demo`   : runs without a real API key, using a
    canned/mocked model response so you can see the full pipeline work.
  - `python src/agent.py`          : interactive mode. Requires NVIDIA_API_KEY
    (from build.nvidia.com) or a self-hosted NIM base_url.

This mirrors the actual integration pattern documented in Get Started with
NVIDIA NIM for LLMs: an OpenAI-compatible client pointed at either NVIDIA's
hosted API catalog or a self-hosted NIM container.
"""
import os
import sys
import json
import argparse
from pathlib import Path

from retriever import NimDocRetriever
from tools import TOOLS, DISPATCH

ROOT = Path(__file__).parent.parent
CORPUS_PATH = ROOT / "corpus.json"

SYSTEM_PROMPT = """You are a developer support assistant for NVIDIA NIM (NVIDIA Inference \
Microservices). You help developers get their NIM deployments and API integrations working.

Rules:
- Ground every factual claim in the provided documentation context. If the context doesn't \
cover something, say so plainly instead of guessing.
- When the developer shares a docker run command or a chat/completions request body, use the \
validate_docker_run or validate_chat_request tool to check it before answering — don't just \
eyeball it.
- Be concise and concrete: give exact flags, env vars, and endpoint paths, not vague advice.
"""


def get_client(base_url: str | None, api_key: str | None):
    from openai import OpenAI
    return OpenAI(
        base_url=base_url or "https://integrate.api.nvidia.com/v1",
        api_key=api_key or os.environ.get("NVIDIA_API_KEY", "not-used"),
    )


def build_context_block(retriever: NimDocRetriever, query: str, top_k: int = 3) -> str:
    hits = retriever.retrieve(query, top_k=top_k)
    if not hits:
        return "(No matching documentation found for this query.)"
    blocks = [f"[Source: {h.source} — {h.heading}]\n{h.text}" for h in hits]
    return "\n\n---\n\n".join(blocks)


def run_agent_turn(client, model: str, retriever: NimDocRetriever, user_message: str, history: list) -> tuple[str, list]:
    context = build_context_block(retriever, user_message)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"Documentation context:\n\n{context}\n\nDeveloper question:\n{user_message}",
    })

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=1024,
        temperature=0.2,
    )
    choice = response.choices[0].message

    # If the model wants to call a tool, execute it and feed the result back.
    if choice.tool_calls:
        messages.append({"role": "assistant", "content": choice.content or "", "tool_calls": choice.tool_calls})
        for tool_call in choice.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            result = DISPATCH[fn_name](**fn_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

        followup = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1024,
            temperature=0.2,
        )
        final_text = followup.choices[0].message.content
    else:
        final_text = choice.content

    new_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": final_text},
    ]
    return final_text, new_history


def run_demo():
    """Runs the retrieval + tool-validation pipeline without a live API key,
    so the mechanics are visible even without NVIDIA credentials."""
    from tools import validate_docker_run

    retriever = NimDocRetriever(CORPUS_PATH)

    print("=" * 70)
    print("DEMO MODE — no API key required. Showing retrieval + tool-calling.")
    print("=" * 70)

    query = "My docker run command isn't working: docker run -it --rm -p 8000:8000 nvcr.io/nim/meta/llama-3.1-8b-instruct:latest"
    print(f"\nDeveloper question:\n{query}\n")

    print("-- RAG retrieval step --")
    hits = retriever.retrieve(query, top_k=3)
    for h in hits:
        print(f"  [{h.score:.3f}] {h.source} :: {h.heading}")

    print("\n-- Tool-calling step (would be triggered by the model) --")
    cmd = "docker run -it --rm -p 8000:8000 nvcr.io/nim/meta/llama-3.1-8b-instruct:latest"
    result = validate_docker_run(cmd)
    print(json.dumps(result, indent=2))

    print("\n-- What the final grounded answer would incorporate --")
    print("  1. Retrieved doc context on Docker Run Parameters + Common Setup Steps")
    print("  2. Structured validator output flagging the missing --gpus and API key")
    print("  3. An LLM call (via NIM's OpenAI-compatible endpoint) to phrase the fix")
    print("\nTo run this end-to-end against a real model, set NVIDIA_API_KEY and run")
    print("without --demo. Get a free key at https://build.nvidia.com/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run without a live API key")
    parser.add_argument("--model", default="meta/llama-3.1-8b-instruct")
    parser.add_argument("--base-url", default=None, help="Override for self-hosted NIM, e.g. http://localhost:8000/v1")
    args = parser.parse_args()

    if args.demo or not os.environ.get("NVIDIA_API_KEY"):
        run_demo()
        return

    retriever = NimDocRetriever(CORPUS_PATH)
    client = get_client(args.base_url, os.environ.get("NVIDIA_API_KEY"))

    print("NIM Developer Support Agent (type 'exit' to quit)\n")
    history: list = []
    while True:
        try:
            user_message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_message.lower() in {"exit", "quit"}:
            break
        if not user_message:
            continue
        answer, history = run_agent_turn(client, args.model, retriever, user_message, history)
        print(f"\nagent> {answer}\n")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
