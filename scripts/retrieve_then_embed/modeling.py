from __future__ import print_function

import gc
import re

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


vllmSizeTokens = re.compile(r"(?:^|[-_.])(\d+)b\b", re.IGNORECASE)


def vllm_billions_from_tag(llm_name):
    """Return sorted unique integer B sizes from tags like -32b, _70b, .8b."""
    return sorted({int(match.group(1)) for match in vllmSizeTokens.finditer(llm_name.lower())})


def vllm_engine_kwargs(
    llm_name,
    gpu_memory_utilization=None,
    tensor_parallel_size=None,
    max_model_len=None,
):
    """
    Map --llm_name to extra vLLM() kwargs using largest \d+b size in tag.
    CLI overrides apply last.
    """
    extras = {}
    sizes = vllm_billions_from_tag(llm_name)
    largest = sizes[-1] if sizes else None
    if largest is not None and largest >= 70:
        extras = {
            "tensor_parallel_size": 2,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.75,
            "max_model_len": 8192,
            "enforce_eager": True,
        }
    elif largest is not None and largest >= 27:
        extras = {
            "gpu_memory_utilization": 0.90,
            "max_model_len": 16000,
        }

    if gpu_memory_utilization is not None:
        extras["gpu_memory_utilization"] = float(gpu_memory_utilization)
    if tensor_parallel_size is not None:
        extras["tensor_parallel_size"] = int(tensor_parallel_size)
    if max_model_len is not None:
        extras["max_model_len"] = int(max_model_len)

    return extras


def build_llm(
    model_id,
    llm_name,
    gpu_memory_utilization=None,
    tensor_parallel_size=None,
    max_model_len=None,
):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    sampling_params = SamplingParams(
        temperature=0, top_p=0.8, repetition_penalty=1.05, max_tokens=32768
    )

    extras = vllm_engine_kwargs(
        llm_name,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
    )
    if not extras:
        print("vLLM: default engine kwargs (model_id only)")
        llm = LLM(model=model_id)
    else:
        print("vLLM extra kwargs: %s" % extras)
        llm = LLM(model=model_id, **extras)

    return tokenizer, sampling_params, llm

