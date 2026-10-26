"""Private local credential storage for cloud AI providers."""

from __future__ import annotations

import json
import os
from pathlib import Path


CONFIG_DIR = Path.home() / ".config" / "h3-studio"
DEEPSEEK_KEY_FILE = CONFIG_DIR / "deepseek_api_key"
MIMO_KEY_FILE = CONFIG_DIR / "mimo_api_key"
MINIMAX_KEY_FILE = CONFIG_DIR / "minimax_api_key"
PROMPT_PROVIDER_FILE = CONFIG_DIR / "prompt_provider"
PROMPT_MODELS_FILE = CONFIG_DIR / "prompt_models"
PROMPT_PROVIDERS = {"auto", "deepseek", "mimo", "minimax"}

# 每个提供商可选的模型。清单只是方便选择，任何合法模型名都能通过
# 自定义输入框使用——云端模型更新很快，写死清单迟早会过期。
PROVIDER_MODEL_DEFAULTS = {
    "deepseek": "deepseek-flash",
    "mimo": "mimo-v2.6-pro",
    "minimax": "MiniMax-M3",
}
PROVIDER_MODEL_CATALOG = {
    "deepseek": ["deepseek-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"],
    "mimo": ["mimo-v2.6-pro", "mimo-v2.5-pro", "mimo-v2.5"],
    "minimax": ["MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.7-highspeed",
                "MiniMax-M2.5", "MiniMax-M2.5-highspeed", "MiniMax-M2", "MiniMax-M2-highspeed"],
}
PROVIDER_ENV_MODELS = {
    "deepseek": "H3_STUDIO_DEEPSEEK_MODEL",
    "mimo": "H3_STUDIO_MIMO_MODEL",
    "minimax": "H3_STUDIO_MINIMAX_MODEL",
}


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


def minimax_key() -> str:
    env_key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        return MINIMAX_KEY_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def minimax_key_source() -> str:
    if os.environ.get("MINIMAX_API_KEY", "").strip():
        return "environment"
    return "saved" if MINIMAX_KEY_FILE.is_file() and minimax_key() else "none"


def save_minimax_key(value: str) -> None:
    key = str(value or "").strip()
    if len(key) < 16 or len(key) > 512 or any(char.isspace() for char in key):
        raise ValueError("MiniMax API Key 格式无效")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(CONFIG_DIR, 0o700)
    temp = CONFIG_DIR / ".minimax_api_key.tmp"
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(key)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp, MINIMAX_KEY_FILE)
        os.chmod(MINIMAX_KEY_FILE, 0o600)
    finally:
        temp.unlink(missing_ok=True)


def remove_saved_minimax_key() -> None:
    MINIMAX_KEY_FILE.unlink(missing_ok=True)


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


def _saved_prompt_models() -> dict:
    try:
        value = json.loads(PROMPT_MODELS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def provider_model(provider: str) -> str:
    """The model this provider should call.

    本机在设置页选过的模型优先，其次是环境变量，最后是内置默认——
    这样既能自己改，又不会因为某个环境变量而改不动。
    """
    name = str(provider or "").strip().lower()
    if name not in PROVIDER_MODEL_DEFAULTS:
        return ""
    saved = str(_saved_prompt_models().get(name) or "").strip()
    if saved:
        return saved
    env_value = os.environ.get(PROVIDER_ENV_MODELS[name], "").strip()
    return env_value or PROVIDER_MODEL_DEFAULTS[name]


def provider_model_source(provider: str) -> str:
    name = str(provider or "").strip().lower()
    if name not in PROVIDER_MODEL_DEFAULTS:
        return "none"
    if str(_saved_prompt_models().get(name) or "").strip():
        return "saved"
    if os.environ.get(PROVIDER_ENV_MODELS[name], "").strip():
        return "environment"
    return "default"


def save_provider_model(provider: str, model: str) -> str:
    name = str(provider or "").strip().lower()
    if name not in PROVIDER_MODEL_DEFAULTS:
        raise ValueError("未知的提示词模型提供商")
    value = str(model or "").strip()
    if not value or len(value) > 64:
        raise ValueError("模型名长度需在 1 到 64 个字符之间")
    if not all(char.isalnum() or char in "-_.:" for char in value):
        raise ValueError("模型名只能包含字母、数字与 - _ . : 符号")
    models = _saved_prompt_models()
    models[name] = value
    CONFIG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(CONFIG_DIR, 0o700)
    temp = CONFIG_DIR / ".prompt_models.tmp"
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(models, output, ensure_ascii=False, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp, PROMPT_MODELS_FILE)
        os.chmod(PROMPT_MODELS_FILE, 0o600)
    finally:
        temp.unlink(missing_ok=True)
    return value
