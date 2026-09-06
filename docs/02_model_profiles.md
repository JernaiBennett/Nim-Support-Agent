# Model Profiles in NVIDIA NIM for LLMs

A NIM model profile defines two things: the runtime the NIM uses, and the criteria NIM should
use to choose those engines. Unique strings based on a hash of the profile contents identify
each profile.

NVIDIA provides optimized model profiles for popular data-center GPU models, different GPU
counts, and specific numeric precisions. Optimized profiles can be pre-compiled TensorRT-LLM
engines or backend runtime configurations across supported backends (`tensorrt_llm`, `vllm`,
`sglang`).

NVIDIA also provides generic model profiles that operate with any NVIDIA GPU (or set of GPUs)
with sufficient memory capacity. Hardware-specific profiles are always preferred over generic
profiles when available.

Model profiles are embedded within the NIM container in a model manifest file, placed by
default at `/opt/nim/etc/default/model_manifest.yaml`.

## Listing Profiles
```
docker run --rm --gpus=all -e NGC_API_KEY=$NGC_API_KEY $IMG_NAME list-model-profiles
```
Example output:
```
MODEL PROFILES
- Compatible with system and runnable:
  - a93a1a6b72643f2b2ee5e80ef25904f4d3f942a87f8d32da9e617eeccfaae04c (tensorrt_llm-A100-fp16-tp2-latency)
  - 751382df4272eafc83f541f364d61b35aed9cce8c7b0c869269cea5a366cd08c (tensorrt_llm-A100-fp16-tp1-throughput)
  - 19031a45cf096b683c4d66fff2a072c0e164a24f19728a58771ebfc4c9ade44f (vllm-fp16-tp2)
  - 8835c31752fbc67ef658b20a9f78e056914fdef0660206d82f252d62fd96064d (vllm-fp16-tp1)
```

## Profile Selection

Set a specific profile with `-e NIM_MODEL_PROFILE=ID`, where ID is a profile ID or profile name
returned by `list-model-profiles`. For example:
`-e NIM_MODEL_PROFILE="tensorrt_llm-A100-fp16-tp1-throughput"`

## Automatic Profile Selection

For LLM-specific NIMs, if no profile is explicitly chosen, NIM selects automatically using this
ordered criteria:

1. **Custom Profiles**: a cached fine-tuned custom profile is selected first if one exists.
2. **Optimized Profiles**: chosen using this sorting logic:
   - **Hardware**: a profile with a `gpu` tag matching the system's hardware is always
     preferred over one that does not.
   - **Backend Engine**: preference order is `tensorrt_llm` > `vllm` > `sglang`.
   - **Precision**: preference order is `MXFP4` > `FP8` > `INT8` > `FP16` > `BF16` > `INT8WO`
     > `NVFP4` > `INT4_AWQ`.
   - **Optimization Target**: latency-optimized profiles are selected over throughput-optimized
     profiles by default.
   - **Tensor Parallelism (TP)**: the highest TP value supported by available GPUs is preferred.
3. **Generic Profiles**: used if the multi-LLM NIM container is deployed or no optimized
   profiles are compatible, following the same backend/precision/TP ordering.

The selection is logged at startup, e.g.:
```
Detected 2 compatible profile(s).
Valid profile: 751382df4272eafc83f541f364d61b35aed9cce8c7b0c869269cea5a366cd08c (tensorrt_llm-A100-fp16-tp1-throughput) on GPUs [0]
Valid profile: 8835c31752fbc67ef658b20a9f78e056914fdef0660206d82f252d62fd96064d (vllm-fp16-tp1) on GPUs [0]
Selected profile: 751382df4272eafc83f541f364d61b35aed9cce8c7b0c869269cea5a366cd08c (tensorrt_llm-A100-fp16-tp1-throughput)
```

## Optimized Profiles vs. Local Build Profiles

Both use TensorRT-LLM. Optimized profiles use GPU-specific TensorRT-LLM options for optimal
throughput/latency and cause NIM to download a pre-compiled engine. Local build profiles use
heuristics to choose a balanced set of options and cause NIM to download raw weights and
compile locally, which can mean longer startup times on first deployment.

## Optimization Targets

`latency` profiles minimize Time to First Token (TTFT) and Inter-Token Latency (ITL).
`throughput` profiles maximize total throughput per GPU, typically using the minimum number of
GPUs required to host the model.

## Quantization

Quantized engines with reduced numeric precision are identified by the numeric format in the
profile name (e.g. `fp8` vs `fp16`). All quantized engines are tested to meet the same accuracy
criteria as `fp16` engines. Because quantization reduces memory requirements and improves
latency/throughput, `fp8` models are chosen by default when available. NIM currently supports
fp8 quantization of HF and Nemotron models.
