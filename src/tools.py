"""
Tool-calling layer: deterministic validators the LLM can invoke.

This is the "agentic" half of the assistant. Rather than letting the LLM
freehand an answer about whether a config is correct, it can call a real
Python function that checks the config against rules taken directly from the
docs corpus. This mirrors what a Solutions Architect actually does: verify a
customer's request/deployment against the platform's real constraints, not
just describe them.
"""
import json
import re
import shlex


def validate_docker_run(command: str) -> dict:
    """
    Checks a `docker run ...` command for launching a NIM container against
    the required flags documented in Get Started with NVIDIA NIM for LLMs.
    """
    issues = []
    warnings = []

    try:
        tokens = shlex.split(command)
    except ValueError as e:
        return {"valid": False, "issues": [f"Could not parse command: {e}"], "warnings": []}

    if not tokens or tokens[0] != "docker" or "run" not in tokens[:2]:
        issues.append("Command does not start with `docker run`.")

    joined = " ".join(tokens)

    required_flags = {
        "--gpus": "Required to expose NVIDIA GPU(s) to the container.",
        "-p": "Required to publish the NIM server port (default 8000) to the host.",
    }
    for flag, why in required_flags.items():
        if flag not in tokens:
            issues.append(f"Missing `{flag}`. {why}")

    # NGC_API_KEY or HF_TOKEN must be present depending on deployment path
    has_ngc_key = "NGC_API_KEY" in joined
    has_hf_token = "HF_TOKEN" in joined
    if not has_ngc_key and not has_hf_token:
        issues.append(
            "No `NGC_API_KEY` or `HF_TOKEN` environment variable found. "
            "NGC/API-Catalog deployments need NGC_API_KEY; HuggingFace-based "
            "multi-LLM NIM deployments need HF_TOKEN."
        )

    if "--runtime=nvidia" not in joined and "--gpus" in joined:
        warnings.append(
            "`--runtime=nvidia` was not found. It's usually paired with `--gpus all` "
            "to guarantee NVIDIA drivers are accessible in the container."
        )

    if "-v" not in tokens and "--volume" not in tokens:
        warnings.append(
            "No cache volume mount (`-v $LOCAL_NIM_CACHE:/opt/nim/.cache`) found. "
            "Without it, the container re-downloads model weights on every restart."
        )

    if "-u" not in tokens:
        warnings.append(
            "No `-u $(id -u)` found. Without it you may hit permission mismatches "
            "writing to the mounted cache directory."
        )

    return {"valid": len(issues) == 0, "issues": issues, "warnings": warnings}


def validate_chat_request(request_body: str) -> dict:
    """
    Checks a JSON request body intended for a NIM /v1/chat/completions or
    /v1/completions call, and confirms it matches the shape that endpoint
    expects (the #1 error documented in Get Started: sending `prompt` to
    the chat endpoint or `messages` to the completions endpoint).
    """
    try:
        body = json.loads(request_body)
    except json.JSONDecodeError as e:
        return {"valid": False, "issues": [f"Not valid JSON: {e}"], "warnings": [], "detected_endpoint": None}

    issues = []
    warnings = []
    has_messages = "messages" in body
    has_prompt = "prompt" in body

    if has_messages and has_prompt:
        issues.append("Body has both `messages` and `prompt`. Use one or the other depending on endpoint.")
        detected_endpoint = None
    elif has_messages:
        detected_endpoint = "/v1/chat/completions"
    elif has_prompt:
        detected_endpoint = "/v1/completions"
    else:
        issues.append("Body has neither `messages` (chat) nor `prompt` (completions). One is required.")
        detected_endpoint = None

    if "model" not in body:
        issues.append("Missing required `model` field.")

    if detected_endpoint == "/v1/chat/completions":
        if not isinstance(body.get("messages"), list) or not body["messages"]:
            issues.append("`messages` must be a non-empty list of {role, content} objects.")
        else:
            for i, m in enumerate(body["messages"]):
                if "role" not in m or "content" not in m:
                    issues.append(f"messages[{i}] is missing `role` or `content`.")

    if "tools" in body and "tool_choice" not in body:
        warnings.append(
            "`tools` is set but `tool_choice` is not. Tool calling needs both parameters; "
            "without `tool_choice` the model may default to never calling a tool."
        )

    if body.get("max_tokens", 0) and body.get("max_tokens") > 32768:
        warnings.append("`max_tokens` is unusually high (>32768) — check this matches your model's context window.")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "detected_endpoint": detected_endpoint,
    }


# Tool schemas in OpenAI/NIM tool-calling format, ready to pass as `tools` in
# a chat.completions.create(..., tools=TOOLS, tool_choice="auto") call.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "validate_docker_run",
            "description": (
                "Validates a `docker run` command intended to launch an NVIDIA NIM "
                "container, checking for required flags (--gpus, -p, API keys) and "
                "common misconfigurations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The full docker run command as a single string."}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_chat_request",
            "description": (
                "Validates a JSON request body intended for a NIM /v1/chat/completions "
                "or /v1/completions call, detecting endpoint mismatches and missing fields."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "request_body": {"type": "string", "description": "The JSON request body as a string."}
                },
                "required": ["request_body"],
            },
        },
    },
]

DISPATCH = {
    "validate_docker_run": validate_docker_run,
    "validate_chat_request": validate_chat_request,
}


if __name__ == "__main__":
    bad_cmd = "docker run -it --rm --name=my-nim -p 8000:8000 nvcr.io/nim/meta/llama-3.1-8b-instruct:latest"
    print("Bad docker run:", json.dumps(validate_docker_run(bad_cmd), indent=2))

    bad_body = '{"model": "meta/llama-3.1-8b-instruct", "prompt": "hi", "messages": [{"role":"user","content":"hi"}]}'
    print("\nAmbiguous request body:", json.dumps(validate_chat_request(bad_body), indent=2))

    good_body = '{"model": "meta/llama-3.1-8b-instruct", "messages": [{"role":"user","content":"hi"}], "max_tokens": 32}'
    print("\nGood request body:", json.dumps(validate_chat_request(good_body), indent=2))
