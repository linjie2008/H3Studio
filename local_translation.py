"""CPU-only Qwen3/CTranslate2 translation of H3 prompts for display."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "data" / "translation" / "qwen3-0.6b-int8"
PROTECTED = re.compile(
    r"(<(?:Picture|Video|Audio|Subject)\s+\d+>|<d>|</d>|<scenetrans>|"
    r"\[\d+(?:\.\d+)?s-\d+(?:\.\d+)?s\])",
    re.IGNORECASE,
)
BASE_INSTRUCTION = (
    "将英文视频镜头描述准确翻译为自然的简体中文，仅输出译文。保留全部人物、动作、场景、"
    "时间顺序和否定要求。特殊标记和数字必须原样保留，不新增或遗漏内容。"
    "不得推断原文未说明的职业、身份、年龄或地点。"
)
ENGLISH_INSTRUCTION = (
    "Translate this Chinese video shot description into natural English. Output only the translation. "
    "Preserve every person, action, scene, time order, negative constraint, reference tag, and number. "
    "Do not invent or omit content. Keep special markers exactly as written."
)
TERMS = {
    "single-take": "一镜到底",
    "one continuous take": "一镜到底",
    "jump cut": "跳切",
    "identity swap": "人物身份互换",
    "occludes the lens": "遮挡镜头",
    "woman": "女人",
    "man": "男人",
}


class Translator:
    def __init__(self) -> None:
        if not (MODEL / "model.bin").is_file():
            raise RuntimeError("Qwen3 本地翻译模型未安装")
        import ctranslate2
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL), local_files_only=True, fix_mistral_regex=True
        )
        self.model = ctranslate2.Generator(
            str(MODEL), device="cpu", compute_type="int8", inter_threads=1, intra_threads=8
        )

    def _translate_piece(self, piece: str, target_language: str = "zh") -> str:
        if not piece.strip() or not re.search(r"[\u4e00-\u9fff]" if target_language == "en" else r"[A-Za-z]", piece):
            return piece
        terms = [f"{english} 译为“{chinese}”" for english, chinese in TERMS.items()
                 if target_language == "zh" and re.search(r"\b" + re.escape(english) + r"\b", piece, re.IGNORECASE)]
        instruction = (ENGLISH_INSTRUCTION if target_language == "en" else BASE_INSTRUCTION)
        instruction += " 术语：" + "；".join(terms) + "。" if terms else ""
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": piece.strip() + " /no_think"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        source = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(prompt, add_special_tokens=False)
        )
        result = self.model.generate_batch(
            [source], max_length=320, sampling_topk=20, sampling_temperature=0.7
        )[0].sequences[0]
        translated = self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(result[len(source):]),
            skip_special_tokens=True,
        ).strip()
        if not translated or translated == piece.strip() or "<think>" in translated:
            raise RuntimeError("Qwen3 未返回有效译文")
        return translated

    def _translate_line(self, line: str, target_language: str = "zh") -> str:
        if not line.strip():
            return line
        markers: dict[str, str] = {}

        def placeholder(match: re.Match) -> str:
            token = f"REFZX{len(markers) + 1:04d}"
            markers[token] = match.group()
            return token

        protected = PROTECTED.sub(placeholder, line)
        sentences = re.split(r"(?<=[.!?。！？])\s+", protected)
        translated = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            # Prompt parts over 300 characters are split at semicolons so the
            # small model does not copy the original English paragraph.
            parts = re.split(r"(?<=;)\s+", sentence) if len(sentence) > 300 else [sentence]
            for part in parts:
                if not part.strip():
                    continue
                output_part = self._translate_piece(part, target_language)
                expected = re.findall(r"REFZX\d{4}", part)
                if any(output_part.count(token) != expected.count(token) for token in set(expected)):
                    # Small language models sometimes simplify away a marker.
                    # Translate around it and place the original marker back
                    # in its original position rather than showing a wrong ID.
                    output_part = "".join(
                        fragment if re.fullmatch(r"REFZX\d{4}", fragment)
                        else self._translate_piece(fragment, target_language)
                        for fragment in re.split(r"(REFZX\d{4})", part)
                    )
                translated.append(output_part)
        output = " ".join(translated)
        for token, original in markers.items():
            if output.count(token) != 1:
                raise RuntimeError("本地译文丢失了参考标记或时间段，请改用 AI 精译")
            output = output.replace(token, original)
        return output

    def translate(self, text: str, target_language: str = "zh") -> str:
        if target_language not in ("zh", "en"):
            raise ValueError("翻译目标语言无效")
        return "\n".join(self._translate_line(line, target_language) for line in text.split("\n")).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--translate", action="store_true")
    args = parser.parse_args()
    try:
        translator = Translator()
        if args.translate:
            print(translator.translate(sys.stdin.read()))
            return 0
        if args.worker:
            request = json.loads(sys.stdin.readline())
            answer = {"translation": translator.translate(
                request["text"], request.get("target_language", "zh"))}
            print(json.dumps(answer, ensure_ascii=False), flush=True)
            return 0
        parser.error("choose --worker or --translate")
    except Exception as error:
        print(f"Qwen3 Translate: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
