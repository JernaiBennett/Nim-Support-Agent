# Function (Tool) Calling with NVIDIA NIM for LLMs

You can connect NIM to external tools and services using function calling (also known as tool
calling). By providing a list of available functions, NIM can output function arguments for
the relevant function(s), which you can execute to augment the prompt with external
information. Function calling is controlled using the `tool_choice` and `tools` request
parameters.

## Prerequisites

For LLM-specific NIMs, do not set tool calling environment variables externally; if the
container supports tool calling, it is enabled automatically. For the multi-LLM NIM container
running on the vLLM backend, tool calling requires two engine arguments:

- `--enable-auto-tool-choice`: allows the model to choose between generating text or calling a
  tool. Required for tool calling.
- `--tool-call-parser <parser>`: a built-in parser name for extracting tool calls from model
  output; must match the model's tool calling format (e.g. `llama3_json` for Llama 3.1/3.3).

Example:
```
docker run --gpus all \
  -e NIM_MODEL_PATH=hf://meta-llama/Llama-3.1-8B-Instruct \
  -p 8000:8000 \
  ${NIM_LLM_MODEL_FREE_IMAGE}:2.0.8 \
  nim-serve --enable-auto-tool-choice --tool-call-parser llama3_json
```

In environments where CLI arguments aren't available (e.g. Kubernetes), pass them via
`NIM_PASSTHROUGH_ARGS`:
```
export NIM_PASSTHROUGH_ARGS="--enable-auto-tool-choice --tool-call-parser llama3_json"
```

If the chat completion response contains an empty `tool_call` field but the function call
appears in the `content` field instead, the response was not post-processed into a tool call
successfully — update the chat template or use a different tool call parser.

## Supported Models

Tool calling is automatically enabled for these LLM-specific NIM containers:
GPT-OSS-20B and GPT-OSS-120B, Llama 3.1 models, Llama 3.2 models, Llama 3.3 models, Mistral
models, Nemotron 3 Nano, Llama Nemotron Nano models (detailed thinking off), Llama Nemotron
Super models (detailed thinking off), Llama Nemotron Ultra models (detailed thinking off).

## Inference Request Parameters for Tools

- `tool_choice`: how the model should choose tools (e.g. `auto`, `required`, `none`, or a
  specific named tool).
- `tools`: the list of tool objects (JSON schema function definitions) available to the model.

## Customizing LLMs for Function Calling

To effectively perform function calling, an LLM must:
- Select the correct function(s)/tool(s) from the available options.
- Extract and populate the correct parameters for each chosen tool from the user's natural
  language query.
- In multi-turn and multi-step use cases, plan and chain multiple tool calls together.

As the number and complexity of tools grows, fine-tuning (including parameter-efficient
approaches like LoRA) can help maintain accuracy and efficiency for smaller models.

Once tool calling is enabled, use the `/v1/chat/completions` endpoint as usual, including the
`tools` and `tool_choice` fields in the request body. The response includes a `tool_calls`
field on the assistant message when the model decides to invoke a tool; your application code
is responsible for executing the named function and returning its result as a follow-up
message with role `tool`.
