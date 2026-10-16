#!/usr/bin/env bash
set -euo pipefail

studio_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$studio_dir/data/translation/qwen3-0.6b-source"
model_dir="$studio_dir/data/translation/qwen3-0.6b-int8"
translation_python="$studio_dir/.venv-translation/bin/python"
comfy_python="${COMFYUI_PYTHON:-$studio_dir/../ComfyUI/venv/bin/python}"
expected_sha256="f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"

if [[ -f "$model_dir/model.bin" ]]; then
    echo "Qwen3 本地翻译模型已安装：$model_dir"
    exit 0
fi
if [[ ! -x "$comfy_python" ]]; then
    echo "找不到用于一次性模型转换的 Python：$comfy_python" >&2
    exit 1
fi
command -v uv >/dev/null || { echo "请先安装 uv" >&2; exit 1; }

if [[ ! -x "$translation_python" ]]; then
    uv venv --python 3.12 "$studio_dir/.venv-translation"
fi
uv pip install --python "$translation_python" 'ctranslate2>=4.8,<5' 'transformers>=4.51,<5' jinja2 huggingface_hub
mkdir -p "$source_dir"
"$studio_dir/.venv-translation/bin/hf" download Qwen/Qwen3-0.6B \
    --include config.json generation_config.json merges.txt tokenizer.json \
    tokenizer_config.json vocab.json --local-dir "$source_dir"

if [[ ! -f "$source_dir/model.safetensors" ]] || \
   [[ "$(sha256sum "$source_dir/model.safetensors" | cut -d' ' -f1)" != "$expected_sha256" ]]; then
    curl -L --fail --retry 5 --retry-delay 2 --connect-timeout 15 \
        -o "$source_dir/model.safetensors" \
        'https://modelscope.cn/models/Qwen/Qwen3-0.6B/resolve/master/model.safetensors'
fi
if [[ "$(sha256sum "$source_dir/model.safetensors" | cut -d' ' -f1)" != "$expected_sha256" ]]; then
    echo "模型校验失败，请重新下载" >&2
    exit 1
fi

translation_packages="$studio_dir/.venv-translation/lib/python3.12/site-packages"
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$translation_packages" "$comfy_python" \
    "$studio_dir/.venv-translation/bin/ct2-transformers-converter" \
    --model "$source_dir" --output_dir "$model_dir" --quantization int8
cp "$source_dir"/{tokenizer.json,tokenizer_config.json,vocab.json,merges.txt,generation_config.json} "$model_dir/"
rm -f "$source_dir/model.safetensors"
echo "Qwen3 本地翻译模型安装完成：$model_dir"
