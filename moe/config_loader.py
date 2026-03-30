"""
moe/config_loader.py
Loads and validates moe_config.yaml.
Falls back to safe defaults if file is missing or malformed.
"""

import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "moe_config.yaml")

_config = None  # cached


def load() -> dict:
    global _config
    if _config is not None:
        return _config

    defaults = {
        "models": {
            "router":   "deepseek-r1:7b",
            "coder":    "qwen2.5-coder:7b",
            "fallback": "llama3:latest",
        },
        "single_agent_models": ["phi3:mini", "llama3:latest", "deepseek-r1:7b"],
        "behaviour": {
            "max_retries":    3,
            "unload_after_use": True,
            "stream_progress":  True,
            "direct_answer_threshold": ["hello", "hi", "help", "who are you"],
        },
        "experts": {},
    }

    try:
        with open(CONFIG_PATH) as f:
            loaded = yaml.safe_load(f) or {}
        # Deep merge with defaults
        for key, val in defaults.items():
            if key not in loaded:
                loaded[key] = val
            elif isinstance(val, dict):
                for k, v in val.items():
                    if k not in loaded[key]:
                        loaded[key][k] = v
        _config = loaded
    except Exception as e:
        print(f"[Config] Could not load moe_config.yaml: {e} — using defaults")
        _config = defaults

    return _config


def get_router_model() -> str:
    cfg = load()
    return cfg["models"].get("router", "deepseek-r1:7b")


def get_coder_model() -> str:
    cfg = load()
    return cfg["models"].get("coder", "qwen2.5-coder:7b")


def get_fallback_model() -> str:
    cfg = load()
    return cfg["models"].get("fallback", "llama3:latest")


def get_single_agent_models() -> list:
    cfg = load()
    return cfg.get("single_agent_models", ["phi3:mini", "llama3:latest", "deepseek-r1:7b"])


def get_max_retries() -> int:
    cfg = load()
    return cfg["behaviour"].get("max_retries", 3)


def get_enabled_experts() -> dict:
    cfg = load()
    return {k: v for k, v in cfg.get("experts", {}).items() if v.get("enabled", True)}


def get_preferred_language() -> str:
    cfg = load()
    return cfg.get("user", {}).get("preferred_language", "python")


def is_direct_answer(task: str) -> bool:
    cfg = load()
    thresholds = cfg["behaviour"].get("direct_answer_threshold", [])
    task_lower = task.lower().strip()
    return any(task_lower.startswith(t) for t in thresholds)
