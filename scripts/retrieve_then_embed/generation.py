from __future__ import print_function

import time

from retrieve_then_embed.prompts import ce_prompts_single_multi, prompt_summarize


def messages_to_prompt(tokenizer, llm_name, messages):
    """Single chat messages list -> one prompt string for vLLM."""
    if "qwen" in llm_name.lower():
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_with_vllm_batch(
    tokenizer, llm, llm_name, messages_list, sampling_params, max_retries=3
):
    """
    Run vLLM on multiple prompts in one call (order preserved).
    Returns list of strings with same length as messages_list.
    """
    for attempt in range(max_retries):
        try:
            texts = [messages_to_prompt(tokenizer, llm_name, msg) for msg in messages_list]
            try:
                outputs = llm.generate(texts, sampling_params, use_tqdm=False)
            except TypeError:
                # use_tqdm is not supported on all vLLM versions
                outputs = llm.generate(texts, sampling_params)
            responses = []
            for request_out in outputs:
                chunk = ""
                for out in request_out.outputs:
                    chunk += out.text
                responses.append(chunk)
            if len(responses) != len(messages_list):
                raise RuntimeError(
                    "vLLM returned %d outputs for %d prompts"
                    % (len(responses), len(messages_list))
                )
            return responses
        except Exception as error:
            print("Batch attempt %d failed: %s" % (attempt + 1, error))
            if attempt < max_retries - 1:
                print("Retrying in 100 seconds...")
                time.sleep(100)
            else:
                print("Max retries reached; returning empty strings for batch.")
                return [""] * len(messages_list)
    return [""] * len(messages_list)


def build_chat_messages(method, document, aspect, llm_name, prompts_yaml):
    """Build OpenAI-style messages for one row."""
    if method == "single":
        single_prompt, _ = ce_prompts_single_multi(document, aspect, prompts_yaml)
        if any(name in llm_name for name in ["gemma", "deepseek"]):
            return [{"role": "user", "content": single_prompt}]
        return [
            {
                "role": "system",
                "content": "You are a precise and efficient extraction model.",
            },
            {"role": "user", "content": single_prompt},
        ]
    if method == "multi":
        _, multi_prompt = ce_prompts_single_multi(document, aspect, prompts_yaml)
        if any(name in llm_name for name in ["gemma", "deepseek"]):
            return [{"role": "user", "content": multi_prompt}]
        return [
            {
                "role": "system",
                "content": "You are a precise and efficient extraction model.",
            },
            {"role": "user", "content": multi_prompt},
        ]
    summary_prompt = prompt_summarize(document, aspect, prompts_yaml)
    if "gemma2" in llm_name.lower():
        return [{"role": "user", "content": summary_prompt}]
    return [
        {
            "role": "system",
            "content": "You are a precise and efficient aspect based summarization model.",
        },
        {"role": "user", "content": summary_prompt},
    ]

