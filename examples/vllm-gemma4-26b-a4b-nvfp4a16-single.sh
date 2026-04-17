#!/bin/bash
# PROFILE: bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16
# DESCRIPTION: Single-node vLLM serve for Gemma4 NVFP4A16 at full 256k context

export VLLM_NVFP4_GEMM_BACKEND=marlin

vllm serve bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16 \
    --served-model-name gemma-4-26b-a4b-it-nvfp4a16 \
    --host 0.0.0.0 \
    --port 8000 \
    --load-format fastsafetensors \
    --enable-prefix-caching \
    --trust-remote-code \
    --quantization modelopt \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --moe-backend marlin \
    --gpu-memory-utilization 0.70 \
    --max-model-len 262144 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 4 \
    -tp 1
