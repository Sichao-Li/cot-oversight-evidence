"""Explicit opt-in Chat Completions transport; credentials never enter output records."""
import math
import os

ENDPOINTS = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
}


def create_client(provider):
    from openai import OpenAI
    url, env = ENDPOINTS[provider]
    key = os.environ.get(env)
    if not key:
        raise ValueError(f"Set {env} in the environment; do not put credentials in a configuration file.")
    return OpenAI(base_url=url, api_key=key, max_retries=0, timeout=180)


def validate_parameters(parameters):
    allowed = {"temperature", "top_p", "max_tokens", "max_completion_tokens", "seed", "reasoning_effort"}
    if not isinstance(parameters, dict) or set(parameters) - allowed:
        raise ValueError("Unsupported decoding field; keys/headers/request overrides are not accepted.")
    if "max_tokens" in parameters and "max_completion_tokens" in parameters:
        raise ValueError("Specify only one output-token limit.")
    for name, value in parameters.items():
        if name in {"max_tokens", "max_completion_tokens", "seed"}:
            valid = type(value) is int and (name == "seed" or value > 0)
        elif name in {"temperature", "top_p"}:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
            if name == "top_p":
                valid = valid and value <= 1
        else:
            valid = isinstance(value, str) and bool(value.strip())
        if not valid:
            raise ValueError(f"Invalid decoding value for {name}; omit unspecified settings.")
    return parameters


def request_vote(client, model, prompt, parameters):
    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                              **validate_parameters(parameters))
    if not response.choices:
        raise ValueError("No completion choice returned.")
    choice = response.choices[0]
    return {"response_id": response.id, "returned_model": response.model,
            "finish_reason": choice.finish_reason, "text": choice.message.content or "",
            "usage": response.usage.model_dump() if response.usage else None}
