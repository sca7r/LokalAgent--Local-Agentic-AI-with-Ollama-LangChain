"""
moe/memory_manager.py
Manages Ollama model loading/unloading to keep only ONE model in RAM at a time.
Uses Ollama's keep_alive=0 to immediately unload a model after use.
"""

import requests
import time

OLLAMA_URL = "http://localhost:11434"

_current_model: str | None = None


def unload_model(model: str) -> None:
    """Immediately unload a model from RAM."""
    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=10,
        )
        print(f"[Memory] Unloaded: {model}")
    except Exception as e:
        print(f"[Memory] Could not unload {model}: {e}")


def unload_current() -> None:
    """Unload whatever model is currently in RAM."""
    global _current_model
    if _current_model:
        unload_model(_current_model)
        _current_model = None


def set_current(model: str) -> None:
    """Track which model is currently loaded."""
    global _current_model
    _current_model = model


def switch_to(model: str) -> None:
    """
    Unload current model and prepare for a new one.
    Ollama auto-loads the new model on first inference call.
    """
    global _current_model
    if _current_model and _current_model != model:
        unload_model(_current_model)
        time.sleep(0.5)  # brief pause for Ollama to release memory
    _current_model = model
    print(f"[Memory] Switched to: {model}")


def get_loaded_models() -> list:
    """Return list of currently loaded models from Ollama."""
    try:
        res = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5)
        data = res.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []
