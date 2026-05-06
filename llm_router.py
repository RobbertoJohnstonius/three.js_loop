"""
llm_router.py — Ollama router for the autonomous game dev loop.

Model assignment (2026-04-25 — qwen2.5-coder:7b promoted to primary coder):
  call_llm()       → llama3.2:latest (131K ctx, ~18-22s)
                     Emergency fallback: gemma2:2b (8K ctx, 4-6s)
                     Roles: critic, playtester, loop critique, planner Stage 1 & 3
  call_llm_coder()          → qwen2.5-coder:7b (32K ctx, ~25-35s) PRIMARY
                               Fallback: llama3.2:latest on failure
                               Roles: coder (interaction tasks), planner Stage 2
                               Rationale: bench R1788 showed 100% VK emission vs 67% for llama3.2
  call_llm_structural_coder() → llama3.2:latest PRIMARY
                               Fallback: qwen2.5-coder:7b on failure
                               Roles: coder (structural tasks — method replacement)
                               Rationale: qwen produces stub-only output (<3 real lines) for
                               structural Python method replacement (R1790 observation)

Installed models (available via `ollama list`):
  qwen2.5-coder:7b     Q4, 32K ctx, 25-35s — PRIMARY coder (100% VK rate on bench)
  llama3.2:latest      Q4, 131K ctx, 18-22s — primary for critic/playtester/planner + coder fallback
  gemma2:2b            Q4, 8K ctx, 4-6s    — emergency fallback for call_llm only
  nomic-embed-text     embeddings (semantic dedup, method retrieval)
  Other: qwen3:1.7b, qwen3:4b, deepseek-r1:1.5b, gemma3:4b, mistral:7b, qwen2.5:7b
"""

import json
import os
import re
import time
import logging
from pathlib import Path

logging.basicConfig(
    handlers=[
        logging.FileHandler("llm_router.log"),
        logging.StreamHandler(),
    ],
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

OLLAMA_URL = "http://127.0.0.1:11434"

# Model assignments
MODEL_FAST   = "gemma2:2b"            # emergency fallback — 8K ctx, use only as last resort
MODEL_SMART  = "llama3.2:latest"      # critic/playtester/planner — 131K ctx, Q4
MODEL_CODER  = "qwen2.5-coder:7b"    # primary coder — 32K ctx, 100% VK rate on bench R1788
MODEL_VISUAL = "qwen2.5-coder:7b"    # visual coder (same model, kept separate for clarity)

# ── OpenRouter cloud provider (opt-in via env var) ───────────────────────────
# Set OPENROUTER_API_KEY to enable. When set, OpenRouter is primary for all
# call_llm* paths; Ollama is fallback. API key is NEVER stored in source.
#
# Recommended models (all < $1/M tokens):
#   General (call_llm):  meta-llama/llama-3.3-70b-instruct  $0.12/M
#   Coder (call_llm_coder): qwen/qwen-2.5-72b-instruct      $0.13/M
#
OPENROUTER_API_KEY    = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL        = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL_SMART  = os.environ.get("OPENROUTER_MODEL_SMART",  "meta-llama/llama-3.3-70b-instruct")
OPENROUTER_MODEL_CODER  = os.environ.get("OPENROUTER_MODEL_CODER",  "qwen/qwen-2.5-72b-instruct")
_OPENROUTER_ENABLED   = bool(OPENROUTER_API_KEY)

# Daily token ceiling — when exceeded, fall back to Ollama for the rest of the day.
# At current pricing ($0.12-0.13/M tokens) this is roughly $0.25-0.30/day max.
DAILY_OPENROUTER_TOKEN_CEILING = int(os.environ.get("OPENROUTER_DAILY_CEILING", "2000000"))

_DAILY_TOKENS_PATH = Path(__file__).parent / "daily_tokens.json"


def _get_daily_tokens() -> dict:
    """Return today's token usage dict, resetting if date has changed."""
    today = time.strftime("%Y-%m-%d")
    try:
        if _DAILY_TOKENS_PATH.exists():
            data = json.loads(_DAILY_TOKENS_PATH.read_text())
            if data.get("date") == today:
                return data
    except Exception:
        pass
    return {"date": today, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _add_daily_tokens(prompt_tokens: int, completion_tokens: int) -> bool:
    """
    Update daily_tokens.json with new usage. Returns True if ceiling still OK,
    False if ceiling exceeded (caller should fall back to Ollama).
    """
    data = _get_daily_tokens()
    data["prompt_tokens"]     += prompt_tokens
    data["completion_tokens"] += completion_tokens
    data["total_tokens"]       = data["prompt_tokens"] + data["completion_tokens"]
    try:
        _DAILY_TOKENS_PATH.write_text(json.dumps(data, indent=2))
    except Exception:
        pass
    if data["total_tokens"] > DAILY_OPENROUTER_TOKEN_CEILING:
        logging.warning(
            f"OpenRouter daily ceiling hit: {data['total_tokens']:,} tokens "
            f"(ceiling={DAILY_OPENROUTER_TOKEN_CEILING:,}) — falling back to Ollama"
        )
        return False
    return True


def get_daily_token_usage() -> dict:
    """Return today's OpenRouter token usage (for status.txt injection)."""
    return _get_daily_tokens()


if _OPENROUTER_ENABLED:
    logging.info(
        f"OpenRouter ENABLED — smart={OPENROUTER_MODEL_SMART} "
        f"coder={OPENROUTER_MODEL_CODER}"
    )

# Tracks which provider won the last coder call
last_coder_provider: str = "ollama"

# Compatibility shims — kept so game_loop.py imports don't break
_groq_banned: set = set()
_cloud_available: bool = False   # always False — cloud removed

# Per-round JSON parse retry counter — reset by game_loop at the start of each round
_json_retry_count: int = 0

def get_json_retry_count() -> int:
    """Return cumulative JSON parse retries this round."""
    return _json_retry_count

def reset_json_retry_count() -> None:
    """Reset retry counter — call at the start of each new round."""
    global _json_retry_count
    _json_retry_count = 0


def _close_json(text: str) -> str:
    """Attempt to close a truncated JSON object by appending missing braces/quotes."""
    text = text.rstrip()
    depth, in_str, escape = 0, False, False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"' and not escape:
            in_str = not in_str
            continue
        if not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
    if in_str:
        text += '"'
    text += "}" * max(0, depth)
    return text


def _extract_json(text: str) -> str:
    """Extract the outermost JSON object from model output."""
    text = text.strip()

    try:
        json.loads(text)
        return text
    except Exception:
        pass

    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    if "```" in text:
        lines = text.splitlines()
        inner, in_block = [], False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            if line.startswith("```") and in_block:
                break
            if in_block:
                inner.append(line)
        fenced = "\n".join(inner).strip()
        s2 = fenced.find("{")
        e2 = fenced.rfind("}")
        if s2 != -1 and e2 != -1 and e2 > s2:
            fenced = fenced[s2:e2 + 1]
        try:
            json.loads(fenced)
            return fenced
        except Exception:
            pass

    fragment = text[text.find("{"):] if "{" in text else text
    repaired = _close_json(fragment)
    try:
        json.loads(repaired)
        return repaired
    except Exception:
        pass

    return text


def _validate_json(text: str) -> str:
    """Validate and return clean JSON string, or raise ValueError."""
    cleaned = _extract_json(text)
    json.loads(cleaned)
    return cleaned


_JSON_HEADER = (
    "IMPORTANT: your entire response MUST be a single valid JSON object. "
    "No prose, no markdown fences, no explanation before or after the JSON.\n\n"
)

_JSON_RETRY_PROMPT = (
    "\n\nYour previous response was not valid JSON. "
    "Reply with ONLY the JSON object — no markdown, no explanation, nothing else."
)


def _call_ollama(model: str, prompt: str, json_mode: bool, timeout: int = 120, label: str = "") -> str:
    """Call Ollama generate endpoint. Returns content string."""
    import requests as req_lib

    base_prompt = (_JSON_HEADER + prompt) if json_mode else prompt

    payload = {
        "model":  model,
        "prompt": base_prompt,
        "stream": False,
    }
    if json_mode:
        payload["format"] = "json"
    t0 = time.time()
    r = req_lib.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    content = r.json().get("response", "")
    latency = time.time() - t0
    tag = f"[{label}] " if label else ""
    logging.info(f"{tag}ollama/{model.split(':')[0]} ok | {latency:.1f}s | {len(content)} chars")

    if json_mode:
        for attempt in range(3):
            try:
                return _validate_json(content)
            except Exception:
                if attempt < 2:
                    global _json_retry_count
                    _json_retry_count += 1
                    logging.warning(f"{tag}ollama JSON parse failed (attempt {attempt+1}/3) — retrying")
                    retry_payload = dict(payload)
                    retry_payload["prompt"] = base_prompt + _JSON_RETRY_PROMPT
                    r2 = req_lib.post(f"{OLLAMA_URL}/api/generate", json=retry_payload, timeout=timeout)
                    r2.raise_for_status()
                    content = r2.json().get("response", "")
                else:
                    raise ValueError(f"Could not extract valid JSON from response: {content[:200]}")
    return content


def _openrouter_ceiling_ok() -> bool:
    """Return False (fall back to Ollama) if daily token ceiling exceeded."""
    data = _get_daily_tokens()
    if data["total_tokens"] > DAILY_OPENROUTER_TOKEN_CEILING:
        logging.warning(
            f"OpenRouter ceiling pre-check: {data['total_tokens']:,} tokens used today "
            f"(ceiling={DAILY_OPENROUTER_TOKEN_CEILING:,}) — skipping OpenRouter this call"
        )
        return False
    return True


def _call_openrouter(
    model: str,
    prompt: str,
    json_mode: bool,
    timeout: int = 120,
    label: str = "",
    system: str | None = None,
) -> str:
    """Call OpenRouter via OpenAI-compatible chat completions endpoint."""
    import requests as req_lib

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if json_mode:
        messages.append({"role": "system", "content": _JSON_HEADER})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/RobbertoJohnstonius/gamedev",
        "X-Title": "Omniville Game Dev Loop",
    }
    payload: dict = {
        "model": model,
        "messages": messages,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    t0 = time.time()
    r = req_lib.post(
        f"{OPENROUTER_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    if not data.get("choices"):
        raise ValueError(f"OpenRouter returned no choices: {str(data)[:200]}")
    content = data["choices"][0]["message"]["content"]
    latency = time.time() - t0
    usage = data.get("usage", {})
    tok_in  = usage.get("prompt_tokens", 0)
    tok_out = usage.get("completion_tokens", 0)
    tag = f"[{label}] " if label else ""
    short_model = model.split("/")[-1]
    logging.info(
        f"{tag}openrouter/{short_model} ok | {latency:.1f}s | {len(content)} chars | "
        f"tokens in={tok_in} out={tok_out}"
    )
    _add_daily_tokens(tok_in, tok_out)

    if json_mode:
        global _json_retry_count
        for attempt in range(3):
            try:
                return _validate_json(content)
            except Exception:
                if attempt < 2:
                    _json_retry_count += 1
                    logging.warning(f"{tag}OpenRouter JSON parse failed (attempt {attempt+1}/3) — retrying")
                    messages_retry = messages + [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": _JSON_RETRY_PROMPT},
                    ]
                    r2 = req_lib.post(
                        f"{OPENROUTER_URL}/chat/completions",
                        headers=headers,
                        json={**payload, "messages": messages_retry},
                        timeout=timeout,
                    )
                    r2.raise_for_status()
                    content = r2.json()["choices"][0]["message"]["content"]
                else:
                    raise ValueError(f"Could not extract valid JSON: {content[:200]}")
    return content


def call_llm(
    prompt: str,
    json_mode: bool = False,
    system: str | None = None,
    max_retries_per_provider: int = 1,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Fast path — llama3.2:latest, falls back to gemma2:2b on failure.

    Roles: critic, playtester, loop critique, planner Stage 1 & 3, visual codex.
    """
    if _OPENROUTER_ENABLED and _openrouter_ceiling_ok():
        try:
            return _call_openrouter(
                OPENROUTER_MODEL_SMART, prompt, json_mode,
                timeout=120, label="or-smart", system=system,
            )
        except Exception as _ore:
            logging.warning(f"OpenRouter smart failed: {str(_ore)[:80]} — falling back to ollama")

    if system:
        prompt = f"{system}\n\n{prompt}"

    try:
        return _call_ollama(MODEL_SMART, prompt, json_mode, timeout=120, label="fast-llama")
    except Exception as _le:
        logging.warning(f"llama3.2 failed: {str(_le)[:80]} — trying gemma2:2b")

    return _call_ollama(MODEL_FAST, prompt, json_mode, timeout=120, label="fast-gemma")


def call_llm_coder(
    prompt: str,
    json_mode: bool = False,
    system: str | None = None,
    max_retries_per_provider: int = 1,
    skip_ollama: bool = False,
) -> str:
    """
    Coder path — qwen2.5-coder:7b (32K ctx) PRIMARY, llama3.2 fallback.
    Bench R1788: qwen2.5-coder 100% VK rate vs llama3.2 67% on 6 interaction tasks.
    gemma2:2b is NOT used here — 8K ctx silently truncates coder prompts.

    Used for: coder (interaction/structural/visual), planner Stage 2.
    """
    global last_coder_provider

    if _OPENROUTER_ENABLED and _openrouter_ceiling_ok():
        _last_or_err = None
        for _attempt in range(2):
            try:
                result = _call_openrouter(
                    OPENROUTER_MODEL_CODER, prompt, json_mode,
                    timeout=240, label="or-coder", system=system,
                )
                last_coder_provider = "openrouter-coder"
                return result
            except Exception as _ore:
                _last_or_err = _ore
                if _attempt == 0:
                    logging.warning(
                        f"OpenRouter coder attempt 1 failed: {str(_ore)[:80]} — retrying in 8s"
                    )
                    time.sleep(8)
        if skip_ollama:
            raise RuntimeError(
                f"OpenRouter coder failed after 2 attempts (ollama disabled): {_last_or_err}"
            )
        logging.warning(
            f"OpenRouter coder failed after 2 attempts: {str(_last_or_err)[:80]} — falling back to ollama"
        )

    if skip_ollama:
        raise RuntimeError("OpenRouter coder unavailable and ollama fallback is disabled")

    if system:
        prompt = f"{system}\n\n{prompt}"

    try:
        result = _call_ollama(MODEL_CODER, prompt, json_mode, timeout=240, label="coder-qwen")
        last_coder_provider = "ollama-qwen"
        return result
    except Exception as _qe:
        logging.warning(
            f"qwen2.5-coder failed: {str(_qe)[:80]} — falling back to llama3.2"
        )
        result = _call_ollama(MODEL_SMART, prompt, json_mode, timeout=240, label="coder-llama-fallback")
        last_coder_provider = "ollama-llama-fallback"
        return result


def call_llm_structural_coder(
    prompt: str,
    json_mode: bool = False,
    system: str | None = None,
) -> str:
    """
    Structural coder path — routes to llama3.2:latest (131K ctx).
    qwen2.5-coder:7b produces stub-only output (<3 real lines) for Python method
    replacement tasks. llama3.2 is used here for structural (non-JSON) code generation.

    Used for: task_type="structural" (World._generate, Needs.update, Game.__init__, etc.)
    """
    # When returning JSON, tell the model explicitly — reduces prose-wrapped JSON
    # that causes parse failures (observed on both OpenRouter and Ollama).
    _json_sys = (
        "Respond with raw JSON only. No markdown, no prose, no code fences. "
        "Output must be valid JSON that can be parsed directly by json.loads()."
        if json_mode else None
    )
    _effective_system = _json_sys or system

    if _OPENROUTER_ENABLED and _openrouter_ceiling_ok():
        # On OpenRouter, use the coder model (qwen-2.5-72b) not the smart model.
        # The stub-only concern was observed with local Ollama qwen2.5-coder:7b (Q4).
        # The 72B OpenRouter version produces full structural output in ~15s vs 300s
        # for llama-3.3-70b-instruct on identical prompts (R2175 observation).
        try:
            return _call_openrouter(
                OPENROUTER_MODEL_CODER, prompt, json_mode,
                timeout=120, label="or-structural", system=_effective_system,
            )
        except Exception as _ore:
            logging.warning(f"OpenRouter structural failed: {str(_ore)[:80]} — falling back to ollama")

    _ollama_system = _effective_system or system
    if _ollama_system:
        prompt = f"{_ollama_system}\n\n{prompt}"
    try:
        result = _call_ollama(MODEL_SMART, prompt, json_mode, timeout=240, label="structural-llama")
        return result
    except Exception as _se:
        logging.warning(
            f"llama3.2 structural failed: {str(_se)[:80]} — falling back to qwen"
        )
        return _call_ollama(MODEL_CODER, prompt, json_mode, timeout=240, label="structural-qwen-fallback")


def call_llm_visual_coder(
    prompt: str,
    json_mode: bool = False,
    system: str | None = None,
) -> str:
    """
    Visual coder path — routes to qwen2.5-coder:7b (purpose-built code model).
    Falls back to call_llm_coder() on failure.

    qwen2.5-coder:7b produces fewer syntax/indentation errors for pygame rendering code.
    Used exclusively for task_type="visual".
    """
    if _OPENROUTER_ENABLED and _openrouter_ceiling_ok():
        try:
            return _call_openrouter(
                OPENROUTER_MODEL_CODER, prompt, json_mode,
                timeout=240, label="or-visual", system=system,
            )
        except Exception as _ore:
            logging.warning(f"OpenRouter visual failed: {str(_ore)[:80]} — falling back to ollama")

    if system:
        prompt = f"{system}\n\n{prompt}"
    try:
        result = _call_ollama(MODEL_VISUAL, prompt, json_mode, timeout=240, label="visual-coder")
        return result
    except Exception as _ve:
        logging.warning(
            f"qwen2.5-coder fallback: {str(_ve)[:80]} — falling back to call_llm_coder"
        )
        return call_llm_coder(prompt, json_mode=json_mode)


# ── T1.3: Instructor-backed structured output ─────────────────────────────────
# Uses OpenRouter when OPENROUTER_API_KEY is set, otherwise Ollama OpenAI-compat.

try:
    import instructor
    from openai import OpenAI as _OpenAI
    if _OPENROUTER_ENABLED:
        _INSTRUCTOR_CLIENT = instructor.from_openai(
            _OpenAI(base_url=OPENROUTER_URL, api_key=OPENROUTER_API_KEY),
            mode=instructor.Mode.JSON,
        )
        _INSTRUCTOR_MODEL_OVERRIDE = OPENROUTER_MODEL_SMART
        logging.info(f"T1.3 Instructor client initialised (OpenRouter — {OPENROUTER_MODEL_SMART})")
    else:
        _INSTRUCTOR_CLIENT = instructor.from_openai(
            _OpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama"),
            mode=instructor.Mode.JSON,
        )
        _INSTRUCTOR_MODEL_OVERRIDE = None
        logging.info("T1.3 Instructor client initialised (Ollama OpenAI-compat endpoint)")
    _INSTRUCTOR_AVAILABLE = True
except Exception as _ie:
    _INSTRUCTOR_AVAILABLE = False
    _INSTRUCTOR_CLIENT = None
    _INSTRUCTOR_MODEL_OVERRIDE = None
    logging.warning(f"T1.3 Instructor unavailable — falling back to call_llm: {_ie}")


def call_llm_structured(
    prompt: str,
    response_model,
    model: str | None = None,
    max_retries: int = 3,
    timeout: int = 120,
):
    """
    T1.3 — Instructor-backed structured LLM call.
    Returns a validated Pydantic model instance.
    Falls back to call_llm(json_mode=True) + manual parse on Instructor failure.
    """
    import json as _json

    _model = model or _INSTRUCTOR_MODEL_OVERRIDE or MODEL_SMART

    if _INSTRUCTOR_AVAILABLE and _INSTRUCTOR_CLIENT is not None:
        try:
            t0 = time.time()
            result = _INSTRUCTOR_CLIENT.chat.completions.create(
                model=_model,
                messages=[{"role": "user", "content": prompt}],
                response_model=response_model,
                max_retries=max_retries,
                timeout=timeout,
            )
            latency = time.time() - t0
            logging.info(
                f"[instructor] {_model.split(':')[0]} ok | {latency:.1f}s | "
                f"schema={response_model.__name__}"
            )
            return result
        except Exception as _e:
            logging.warning(
                f"T1.3 Instructor call failed ({_model}): {str(_e)[:100]} — "
                f"falling back to call_llm"
            )

    raw = call_llm(prompt, json_mode=True)
    data = _json.loads(raw)
    return response_model.model_validate(data)


# ── Vision call (Groq primary, OpenRouter fallback) ───────────────────────────
#
# Primary: Groq llama-4-scout-17b (~6s, free tier) — multimodal, fast
# Fallback: OpenRouter qwen2.5-vl-72b-instruct (24s, $0.4/M tokens)
# Override primary model via GROQ_MODEL_VISION env var.

GROQ_API_KEY        = os.environ.get("GROQ_API_KEY", "")
GROQ_URL            = "https://api.groq.com/openai/v1"
GROQ_MODEL_VISION   = os.environ.get("GROQ_MODEL_VISION", "meta-llama/llama-4-scout-17b-16e-instruct")
_GROQ_VISION_ENABLED = bool(GROQ_API_KEY)

OPENROUTER_MODEL_VISION = os.environ.get(
    "OPENROUTER_MODEL_VISION",
    "qwen/qwen2.5-vl-72b-instruct",  # 72B fallback
)


def _call_groq_vision(prompt: str, b64: str, timeout: int = 30) -> str:
    """Send prompt + base64 image to Groq vision. Raises on failure."""
    import requests as req_lib
    r = req_lib.post(
        f"{GROQ_URL}/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL_VISION,
            "max_tokens": 1200,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}],
        },
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    if not data.get("choices"):
        raise ValueError(f"Groq vision returned no choices: {str(data)[:200]}")
    return data["choices"][0]["message"]["content"]


def call_llm_vision(
    prompt: str,
    image_path,
    model: str | None = None,
    timeout: int = 60,
    json_mode: bool = False,
) -> str:
    """
    Send prompt + image to vision LLM. Tries Groq first (~6s), falls back to OpenRouter (~24s).
    image_path: str or Path — PNG/JPEG file on disk.
    Returns response text, or "" on failure.
    """
    import base64
    from pathlib import Path as _Path

    try:
        image_bytes = _Path(image_path).read_bytes()
    except Exception as e:
        logging.warning(f"call_llm_vision: could not read image {image_path}: {e}")
        return ""

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    text_block = (_JSON_HEADER + prompt) if json_mode else prompt

    # ── Try Groq first ────────────────────────────────────────────────────────
    if _GROQ_VISION_ENABLED and not model:
        t0 = time.time()
        try:
            content = _call_groq_vision(text_block, b64, timeout=30)
            logging.info(f"[vision] groq/{GROQ_MODEL_VISION.split('/')[-1]} ok | {time.time()-t0:.1f}s | {len(content)} chars")
            if json_mode:
                try:
                    return _validate_json(content)
                except Exception:
                    logging.warning("[vision] Groq JSON extraction failed — returning raw")
            return content
        except Exception as e:
            logging.warning(f"[vision] Groq failed ({str(e)[:80]}) — falling back to OpenRouter")

    # ── Fallback: OpenRouter ──────────────────────────────────────────────────
    vision_model = model or OPENROUTER_MODEL_VISION

    if not _OPENROUTER_ENABLED:
        logging.warning("call_llm_vision: neither Groq nor OpenRouter available — skipping vision")
        return ""

    import requests as req_lib
    payload = {
        "model": vision_model,
        "max_tokens": 1200,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": text_block},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
    }
    t0 = time.time()
    try:
        r = req_lib.post(
            f"{OPENROUTER_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/threejs-loop",
                "X-Title": "threejs-loop",
            },
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        if not data.get("choices"):
            raise ValueError(f"OpenRouter vision returned no choices: {str(data)[:200]}")
        content = data["choices"][0]["message"]["content"]
        logging.info(f"[vision] or/{vision_model.split('/')[-1]} ok | {time.time()-t0:.1f}s | {len(content)} chars")
        if json_mode:
            try:
                return _validate_json(content)
            except Exception:
                logging.warning("[vision] OpenRouter JSON extraction failed — returning raw")
        return content
    except Exception as e:
        logging.warning(f"call_llm_vision failed (OpenRouter/{vision_model}): {str(e)[:100]}")
        return ""


# ── Quick smoke-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing llm_router (local-only)...")
    r1 = call_llm('Reply with: {"ok": true}', json_mode=True)
    print("Fast path:", r1)
    r2 = call_llm_coder('Reply with: {"ok": true}', json_mode=True)
    print("Coder path:", r2)
