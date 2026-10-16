"""Private local credential storage for cloud AI providers."""

from __future__ import annotations

import os
from pathlib import Path


CONFIG_DIR = Path.home() / ".config" / "h3-studio"
DEEPSEEK_KEY_FILE = CONFIG_DIR / "deepseek_api_key"
MIMO_KEY_FILE = CONFIG_DIR / "mimo_api_key"
PROMPT_PROVIDER_FILE = CONFIG_DIR / "prompt_provider"
PROMPT_PROVIDERS = {"auto", "deepseek", "mimo"}


def deepseek_key() -> str:
    env_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        return DEEPSEEK_KEY_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def deepseek_key_source() -> str:
    if os.environ.get("DEEPSEEK_API_KEY", "").strip():
        return "environment"
    return "saved" if DEEPSEEK_KEY_FILE.is_file() and deepseek_key() else "none"


def save_deepseek_key(value: str) -> None:
    key = str(value or "").strip()
    if len(key) < 16 or len(key) > 512 or any(char.isspace() for char in key):
        raise ValueError("DeepSeek API Key 格式无效")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(CONFIG_DIR, 0o700)
    temp = CONFIG_DIR / ".deepseek_api_key.tmp"
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(key)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp, DEEPSEEK_KEY_FILE)
        os.chmod(DEEPSEEK_KEY_FILE, 0o600)
    finally:
        temp.unlink(missing_ok=True)


def remove_saved_deepseek_key() -> None:
    DEEPSEEK_KEY_FILE.unlink(missing_ok=True)


def mimo_key() -> str:
    env_key = os.environ.get("MIMO_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        return MIMO_KEY_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def mimo_key_source() -> str:
    if os.environ.get("MIMO_API_KEY", "").strip():
        return "environment"
    return "saved" if MIMO_KEY_FILE.is_file() and mimo_key() else "none"


def save_mimo_key(value: str) -> None:
    key = str(value or "").strip()
    if len(key) < 16 or len(key) > 512 or any(char.isspace() for char in key):
        raise ValueError("小米 MiMo API Key 格式无效")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(CONFIG_DIR, 0o700)
    temp = CONFIG_DIR / ".mimo_api_key.tmp"
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(key)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp, MIMO_KEY_FILE)
        os.chmod(MIMO_KEY_FILE, 0o600)
    finally:
        temp.unlink(missing_ok=True)


def remove_saved_mimo_key() -> None:
    MIMO_KEY_FILE.unlink(missing_ok=True)


def prompt_provider() -> str:
    """Return the user's cloud prompt model preference; auto preserves fallback."""
    value = os.environ.get("H3_STUDIO_PROMPT_PROVIDER", "").strip().lower()
    if not value:
        try:
            value = PROMPT_PROVIDER_FILE.read_text(encoding="utf-8").strip().lower()
        except FileNotFoundError:
            value = "auto"
    return value if value in PROMPT_PROVIDERS else "auto"


def save_prompt_provider(value: str) -> str:
    provider = str(value or "").strip().lower()
    if provider not in PROMPT_PROVIDERS:
        raise ValueError("提示词模型选择无效")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(CONFIG_DIR, 0o700)
    temp = CONFIG_DIR / ".prompt_provider.tmp"
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(provider)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp, PROMPT_PROVIDER_FILE)
        os.chmod(PROMPT_PROVIDER_FILE, 0o600)
    finally:
        temp.unlink(missing_ok=True)
    return provider
