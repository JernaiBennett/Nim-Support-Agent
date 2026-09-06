# Get Started with NVIDIA NIM for LLMs

## Prerequisites

### GPU
NVIDIA NIM for LLMs should, but is not guaranteed to, run on any NVIDIA GPU, as long as the GPU
has sufficient memory, or on multiple, homogeneous NVIDIA GPUs with sufficient aggregate memory
and CUDA compute capability > 7.0 (8.0 for bfloat16).

Approximate memory guidelines:
- 5-10 GB for OS and other processes
- 16 GB for Docker (shared memory required in multi-GPU, non-NVLink cases)
- # model parameters * 2 GB of memory
  - Llama 8B: ~15 GB
  - Llama 70B: ~131 GB
  - Mistral 7B Instruct v0.3: ~14 GB
  - Mixtral 8x7B Instruct v0.1: ~88 GB

### Software
- A Linux operating system (Ubuntu 22.04 or later recommended) with glibc >= 2.35
- NVIDIA Driver release 580 or later
- Docker >= 23.0.1
- CUDA 13.0
- NVIDIA GPU(s) with sufficient memory; homogeneous multi-GPU systems with tensor parallelism supported

### NIM Container Access
To download and deploy a NIM container, join the free NVIDIA Developer Program or obtain an
NVIDIA AI Enterprise license. To join the Developer Program: go to the NVIDIA API Catalog
(build.nvidia.com), find a model, select the Deploy tab, click "Get API key", and sign in.

## Launch NVIDIA NIM for LLMs

Three deployment approaches:
1. API Catalog: use pre-built, optimized models and LLM-specific NIM containers directly from
   NVIDIA's API catalog.
2. NGC: use pre-built, optimized models and LLM-specific NIM containers from the NGC registry.
3. HuggingFace or Local Disk: use the multi-LLM compatible NIM container to deploy supported
   HuggingFace or local models.

### Common Setup Steps

**Generate an API Key (API Catalog deployments):**
1. Navigate to the API Catalog (build.nvidia.com).
2. Select a model.
3. Select an Input option.
4. Select "Get API Key" and sign in if prompted.
5. Select "Generate Key".
6. Copy your key and store it securely. Do not share it.

**Generate an API Key (NGC deployments):**
An NGC Personal API key is required. Generate it on the Setup API Keys page
(org.ngc.nvidia.com/setup/api-keys). Legacy API keys are not supported by NVIDIA NIM for LLMs;
always use a Personal API key. Select at least "NGC Catalog" from Services Included.

**Export the API key:**
```
export NGC_API_KEY=VALUE
```

**Docker Login (NGC deployments):**
```
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```
Use `$oauthtoken` as the username and `NGC_API_KEY` as the password.

### Option 2: Deploy from NGC (LLM-specific NIM)

List available NIMs:
```
ngc registry image list --format_type ascii nvcr.io/nim/*
```

Launch example (llama-3.1-8b-instruct):
```
export CONTAINER_NAME=llama-3.1-8b-instruct
export Repository=nim/meta/llama-3.1-8b-instruct
export TAG=latest
export IMG_NAME="nvcr.io/$Repository:$TAG"
export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE"
chmod -R a+w "$LOCAL_NIM_CACHE"

docker run -it --rm --name=$CONTAINER_NAME \
  --runtime=nvidia \
  --gpus all \
  --shm-size=16GB \
  -e NGC_API_KEY=$NGC_API_KEY \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -u $(id -u) \
  -p 8000:8000 \
  $IMG_NAME
```

### Option 3: HuggingFace or Local Disk (Multi-LLM NIM)

Set `NIM_MODEL_NAME` to a HuggingFace repo (`hf://org/model-name`, requires `HF_TOKEN`) or a
local model folder path. Supported checkpoint formats: HuggingFace full precision/quantized
safetensors, HuggingFace GGUF, unified HuggingFace quantized models, TRTLLM pre-built
checkpoints, and TRTLLM pre-built engines.

```
export CONTAINER_NAME=LLM-NIM
export Repository=nim/nvidia/llm-nim
export TAG=latest
export HF_TOKEN=hf_xxxxxx
export IMG_NAME="nvcr.io/$Repository:$TAG"
export NIM_MODEL_NAME=hf://meta-llama/Llama-3.1-8B-Instruct
export NIM_SERVED_MODEL_NAME=meta/llama-3.1-8b-instruct
export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE"
chmod -R a+w "$LOCAL_NIM_CACHE"

docker run -it --rm --name=$CONTAINER_NAME \
  --runtime=nvidia \
  --gpus all \
  --shm-size=16GB \
  -e HF_TOKEN=$HF_TOKEN \
  -e NIM_MODEL_NAME=$NIM_MODEL_NAME \
  -e NIM_SERVED_MODEL_NAME=$NIM_SERVED_MODEL_NAME \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -u $(id -u) \
  -p 8000:8000 \
  $IMG_NAME
```

## Docker Run Parameters

| Flag | Description |
| --- | --- |
| `-it` | interactive + tty |
| `--rm` | delete container after it stops |
| `--name` | container name for bookkeeping |
| `--runtime=nvidia` | ensures NVIDIA drivers are accessible in the container |
| `--gpus all` | expose all NVIDIA GPUs inside the container |
| `--shm-size=16GB` | host memory for multi-GPU communication (not required for single GPU or NVLink) |
| `-e NGC_API_KEY` | token to download models/resources from NGC |
| `-v "$LOCAL_NIM_CACHE:/opt/nim/.cache"` | mount a cache dir so downloaded models are reused |
| `-u $(id -u)` | avoid permission mismatches when downloading to local cache |
| `-p 8000:8000` | forward the NIM server port to the host |

## Run Inference

The server exposes an OpenAI-compatible API. Check readiness before use:
```
curl http://0.0.0.0:8000/v1/health/ready
```
List available models:
```
curl -X GET 'http://0.0.0.0:8000/v1/models'
```

### Completions Request (base models)
```
curl -X 'POST' 'http://0.0.0.0:8000/v1/completions' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "meta/llama-3.1-8b-instruct",
      "prompt": "Once upon a time",
      "max_tokens": 64
    }'
```

Using the OpenAI Python library:
```python
from openai import OpenAI
client = OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="not-used")
response = client.completions.create(
    model="meta/llama-3.1-8b-instruct",
    prompt="Once upon a time",
    max_tokens=16,
    stream=False
)
print(response.choices[0].text)
```

### Chat Completions Request (chat/instruct models)
```
curl -X 'POST' 'http://0.0.0.0:8000/v1/chat/completions' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "meta/llama-3.1-8b-instruct",
      "messages": [
          {"role":"user","content":"Hello! How are you?"},
          {"role":"assistant","content":"Hi! I am quite well, how can I help you today?"},
          {"role":"user","content":"Can you write me a song?"}
      ],
      "max_tokens": 32
    }'
```

Common error: sending a `prompt` field to `/v1/chat/completions`, or a `messages` field to
`/v1/completions`, returns a 400 BadRequestError naming the missing field. Verify you are
using the correct endpoint for the request shape you built.

## Stopping the Container
```
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME
```
