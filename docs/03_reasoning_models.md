# Use Reasoning Models with NVIDIA NIM for LLMs

NIM for LLMs supports deploying reasoning models designed to generate detailed, step-by-step
thought processes. These models are post-trained using unique system prompts to support two
modes: chain-of-thought responses, and concise responses. A single model can toggle between
the two by changing the request; no additional scaffolding is required.

## Reasoning Mode

Reasoning mode is controlled by the system prompt or chat template keyword arguments,
depending on the model. When controlled by the system prompt, the system prompt must be the
first message in the conversation.

### Detailed Thinking Prompt (Llama 3.3 Nemotron Super 49B V1, Llama 3.1 Nemotron Ultra 253B V1)

| System Prompt | Description | Recommended Settings |
| --- | --- | --- |
| `detailed thinking on` | Long chain-of-thought style responses with explicit thinking tokens. | temperature=0.6, top_p=0.95 |
| `detailed thinking off` | Concise responses without extended chain-of-thought. | temperature=0 |

Example (detailed thinking on):
```python
from openai import OpenAI
client = OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="not-used")

messages = [
    {"role": "system", "content": "detailed thinking on"},
    {"role": "user", "content": "How many 'r's are in 'strawberry'?"}
]

chat_response = client.chat.completions.create(
    model="nvidia/llama-3.3-nemotron-super-49b-v1",
    messages=messages,
    max_tokens=32768,
    stream=False,
    temperature=0.6,
    top_p=0.95
)
print(chat_response.choices[0].message)
```

### No Think Prompt (Llama 3.3 Nemotron Super 49B V1.5)

Detailed reasoning is on by default for this model. Use `/no_think` as the system prompt to
get concise responses without thinking tokens (recommended temperature=0).

### Chat Template Keyword Arguments (Nemotron 3 Nano, Nemotron 3 Super)

Use `enable_thinking` in `chat_template_kwargs` (default `True`). Set to `False` for concise
responses without extended chain-of-thought:
```python
extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
chat_response = client.chat.completions.create(
    model="nvidia/nemotron-3-nano",
    messages=[{"role": "user", "content": "Tell me a story."}],
    max_tokens=32768,
    stream=False,
    temperature=0,
    extra_body=extra_body,
)
```

## Parallel Reasoning

Some models (Nemotron 3 Nano) support parallel reasoning traces for difficult problems via
`parallel_reasoning_mode` in `chat_template_kwargs`: `none` (default), `low`, `medium`, or
`heavy`. Parallel reasoning mode is incompatible with tool calling.

## Reasoning Effort

The `reasoning_effort` parameter controls how much computational effort the model spends on
reasoning. Supported only by the Chat Completions API for GPT-OSS-20B and GPT-OSS-120B when
deployed using the multi-LLM NIM, and requires `-e NIM_REASONING_PARSER=openai_gptoss` at
deployment.

| Reasoning Effort | Description | Use Case |
| --- | --- | --- |
| `low` | Faster responses, less reasoning depth. | Quick or time-sensitive queries. |
| `medium` | Balanced reasoning and speed. | General-purpose reasoning tasks. |
| `high` | Maximum reasoning depth. | Complex problem-solving. |

Example request:
```
curl --location 'http://0.0.0.0:8000/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --data '{
    "messages": [{"role": "user", "content": "What is the role of AI in medicine?"}],
    "model": "openai/gpt-oss-120b",
    "reasoning_effort": "high",
    "stream": false
  }'
```
