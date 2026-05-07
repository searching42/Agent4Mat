#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from urllib.parse import urlparse


def _default_config_path() -> Path:
    return Path.home() / ".openclaw" / "agents" / "main" / "agent" / "models.json"


def _load_json(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("models.json root must be object")
    return obj


def _providers(payload: dict) -> dict:
    providers = payload.get("providers")
    if not isinstance(providers, dict) or not providers:
        raise ValueError("models.json missing non-empty providers object")
    return providers


def _pick_provider(providers: dict, provider_name: str) -> tuple[str, dict]:
    if provider_name:
        provider = providers.get(provider_name)
        if not isinstance(provider, dict):
            raise ValueError(f"provider not found: {provider_name}")
        return provider_name, provider
    # Keep deterministic default.
    first_name = sorted(providers.keys())[0]
    provider = providers[first_name]
    if not isinstance(provider, dict):
        raise ValueError(f"provider config invalid: {first_name}")
    return first_name, provider


def _pick_model_id(provider: dict, model_id_override: str) -> str:
    if model_id_override:
        return model_id_override
    models = provider.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("provider.models is empty")
    first = models[0]
    if not isinstance(first, dict):
        raise ValueError("provider.models[0] is invalid")
    model_id = str(first.get("id") or first.get("model") or "").strip()
    if not model_id:
        raise ValueError("provider.models[0] missing id/model")
    return model_id


def _derive_chat_path(base_url: str) -> str:
    path = (urlparse(base_url).path or "").rstrip("/")
    # If base url already ends with /v1, append plain /chat/completions.
    if path.endswith("/v1"):
        return "/chat/completions"
    return "/v1/chat/completions"


def _build_env_map(provider_name: str, provider: dict, model_id: str) -> dict:
    base_url = str(provider.get("baseUrl") or provider.get("base_url") or "").strip()
    api_key = str(provider.get("apiKey") or provider.get("api_key") or "").strip()
    if not base_url:
        raise ValueError(f"provider {provider_name} missing baseUrl")
    if not api_key:
        raise ValueError(f"provider {provider_name} missing apiKey")
    chat_path = _derive_chat_path(base_url)
    return {
        "OLED_AGENT_LLM_BACKEND": "openai_compat",
        "OLED_AGENT_LLM_MODEL": model_id,
        "OLED_AGENT_LLM_API_KEY": api_key,
        "OLED_AGENT_LLM_BASE_URL": base_url,
        "OLED_AGENT_LLM_CHAT_COMPLETIONS_PATH": chat_path,
        "OLED_AGENT_LLM_AUTH_HEADER": "Authorization",
        "OLED_AGENT_LLM_AUTH_SCHEME": "Bearer",
    }


def _as_exports(env_map: dict) -> str:
    lines = [
        "unset OLED_AGENT_LLM_PLANNER_CMD",
        "unset OLED_AGENT_LLM_EXTRA_HEADERS_JSON",
        "unset OLED_AGENT_LLM_DISABLE_RESPONSE_FORMAT",
    ]
    for key in sorted(env_map.keys()):
        lines.append(f"export {key}={shlex.quote(str(env_map[key]))}")
    return "\n".join(lines) + "\n"


def _as_dotenv(env_map: dict) -> str:
    lines = []
    for key in sorted(env_map.keys()):
        val = str(env_map[key]).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'{key}="{val}"')
    return "\n".join(lines) + "\n"


def _print_provider_list(providers: dict) -> int:
    rows = []
    for name in sorted(providers.keys()):
        provider = providers.get(name)
        if not isinstance(provider, dict):
            continue
        models = provider.get("models")
        model_id = ""
        if isinstance(models, list) and models and isinstance(models[0], dict):
            model_id = str(models[0].get("id") or models[0].get("model") or "").strip()
        rows.append(
            {
                "provider": name,
                "base_url": str(provider.get("baseUrl") or provider.get("base_url") or ""),
                "default_model_id": model_id,
                "has_api_key": bool(provider.get("apiKey") or provider.get("api_key")),
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export oled-agent LLM env vars from OpenClaw models.json")
    p.add_argument("--config", default=str(_default_config_path()), help="Path to OpenClaw models.json")
    p.add_argument("--provider", default="", help="Provider key in models.json providers object")
    p.add_argument("--model-id", default="", help="Override model id")
    p.add_argument(
        "--format",
        default="exports",
        choices=["exports", "dotenv"],
        help="Output format",
    )
    p.add_argument("--list-providers", action="store_true", help="List providers and exit")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"[FAIL] config not found: {config_path}", file=sys.stderr)
        return 2
    try:
        payload = _load_json(config_path)
        providers = _providers(payload)
        if args.list_providers:
            return _print_provider_list(providers)
        provider_name, provider = _pick_provider(providers, args.provider)
        model_id = _pick_model_id(provider, args.model_id)
        env_map = _build_env_map(provider_name, provider, model_id)
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    if args.format == "dotenv":
        sys.stdout.write(_as_dotenv(env_map))
    else:
        sys.stdout.write(_as_exports(env_map))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
