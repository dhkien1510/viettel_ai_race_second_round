"""Unified chat() across providers. API keys are read from environment variables
so you just export the key and run. SDKs are imported lazily with a friendly
message if missing.

Env vars:
    OPEN_ROUTER_API     (provider: openrouter — one key, every model)
                        [OPENROUTER_API_KEY cũng được chấp nhận]
    OPENAI_API_KEY      (provider: openai, and openai-compatible/self-host)
    OPEN_CODE_GO        (fallback key for openai-compatible when OPENAI_API_KEY
                         is unset — https://opencode.ai/go subscription gateway)
    ANTHROPIC_API_KEY   (provider: anthropic)
    GOOGLE_API_KEY      (provider: google/gemini)   [or GEMINI_API_KEY]

Providers:
    openrouter  -> chat.completions @ openrouter.ai. Model ids carry the vendor
                   prefix. This project uses exactly two, on purpose:
                       generate : "google/gemini-2.5-flash"
                       judge    : "qwen/qwen-2.5-72b-instruct"
                   A DIFFERENT vendor judges than generates — same-vendor judges
                   correlate their mistakes and rubber-stamp each other.

                   NOTE: the Claude Opus audit pass (25% sample) is NOT an API
                   call and no anthropic provider is needed for it. It is done
                   in-session by Claude Code reading data/synth/pending/ directly.
                   Do not add an Anthropic key for the synth pipeline.
    openai      -> chat.completions (also 'openai-compatible' with base_url for
                   self-hosted vLLM / Together / OpenCode Go / etc.)
    anthropic   -> messages
    google      -> google-generativeai
"""

from __future__ import annotations

import os

from .env import load_dotenv

# auto-load repo-root .env so keys are available just by filling in .env
load_dotenv()

# Some openai-compatible gateways sit behind Cloudflare and reject the SDK's
# default User-Agent (error code 1010) — send a browser-like one everywhere.
_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def _need(pkg: str, provider: str):
    raise SystemExit(
        f"Provider '{provider}' cần package '{pkg}'. Cài: pip install {pkg}"
    )


def chat(provider: str, model: str, system: str, user: str,
         temperature: float = 0.7, max_tokens: int = 2048,
         api_key: str = None, base_url: str = None) -> str:
    provider = (provider or "").lower()

    if provider == "openrouter":
        try:
            from openai import OpenAI
        except Exception:
            _need("openai", provider)
        # .env của repo này dùng tên OPEN_ROUTER_API; chấp nhận cả tên chuẩn của
        # OpenRouter để người khác clone về không phải đổi gì.
        key = (api_key or os.getenv("OPEN_ROUTER_API")
               or os.getenv("OPENROUTER_API_KEY"))
        if not key:
            raise SystemExit(
                "Thiếu key OpenRouter. Đặt vào .env ở gốc repo:\n"
                "    OPEN_ROUTER_API=sk-or-...\n"
                "(lấy key ở https://openrouter.ai/keys)"
            )
        client = OpenAI(
            api_key=key,
            base_url=base_url or "https://openrouter.ai/api/v1",
            default_headers={"User-Agent": _BROWSER_UA},
        )
        r = client.chat.completions.create(
            model=model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        # OpenRouter surfaces upstream failures as a 200 with an `error` field and
        # no choices — surface it instead of silently emitting an empty note that
        # the validator would then reject as "0 khái niệm".
        #
        # RuntimeError, KHÔNG PHẢI SystemExit. SystemExit kế thừa BaseException,
        # nên nó ĐI XUYÊN QUA mọi `except Exception` mà caller viết để chống lỗi
        # từng lượt gọi. Đã trả giá: một lần OpenRouter định tuyến nhầm provider
        # (lỗi THOÁNG QUA — chính model đó thử lại thì chạy) đã giết cả mẻ judge
        # 530 note và dời oan 143 note tốt sang rejected/.
        # Lỗi MỘT lượt gọi là chuyện thường; nó phải bắt được, không được thoát ra.
        if not getattr(r, "choices", None):
            raise RuntimeError(f"OpenRouter trả về lỗi cho model {model!r}: "
                               f"{getattr(r, 'error', None) or r}")
        return r.choices[0].message.content or ""

    if provider in ("openai", "openai-compatible", "vllm", "together", "self-host"):
        try:
            from openai import OpenAI
        except Exception:
            _need("openai", provider)
        key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_CODE_GO")
        if not key and not base_url:
            raise SystemExit("Thiếu OPENAI_API_KEY (export trước khi chạy).")
        client = OpenAI(
            api_key=key, base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            default_headers={"User-Agent": _BROWSER_UA},
        )
        r = client.chat.completions.create(
            model=model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return r.choices[0].message.content or ""

    if provider == "anthropic":
        try:
            from anthropic import Anthropic
        except Exception:
            _need("anthropic", provider)
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("Thiếu ANTHROPIC_API_KEY (export trước khi chạy).")
        client = Anthropic(api_key=key)
        r = client.messages.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")

    if provider in ("google", "gemini"):
        try:
            import google.generativeai as genai
        except Exception:
            _need("google-generativeai", provider)
        key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise SystemExit("Thiếu GOOGLE_API_KEY / GEMINI_API_KEY (export trước khi chạy).")
        genai.configure(api_key=key)
        m = genai.GenerativeModel(model, system_instruction=system)
        r = m.generate_content(
            user, generation_config={"temperature": temperature,
                                     "max_output_tokens": max_tokens})
        return r.text or ""

    raise SystemExit(f"Provider không hỗ trợ: {provider!r} "
                     f"(dùng: openai | anthropic | google | openai-compatible)")
