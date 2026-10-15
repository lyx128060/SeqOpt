"""Verify the exact Responses API path used by the seq_align proposer."""

from __future__ import annotations

import os
import sys
import argparse


def main() -> int:
    from openai import OpenAI
    from seq_align.c_patch_loop.context import plan_prompt
    from seq_align.c_patch_loop.models import ContextPackRecord

    parser = argparse.ArgumentParser(description="Verify the Responses API used by seq_align proposals")
    parser.add_argument("--context-pack", default=None,
                        help="optional saved context-pack JSON to test a real proposal-sized request")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_API_BASE", "http://127.0.0.1:8318/v1")
    model = os.getenv("LLM_MODEL", "gpt-5.6-sol")
    timeout = int(os.getenv("MODEL_TIMEOUT", "1200"))
    effort = os.getenv("SEQ_ALIGN_MODEL_REASONING_EFFORT", "medium")
    if not api_key:
        print("OPENAI_API_KEY or LLM_API_KEY is not set", file=sys.stderr)
        return 2

    instructions = "You output only valid JSON for one bounded C patch plan."
    input_text = (
        "Return only this exact JSON object and no markdown: "
        '{\"status\":\"ok\"}'
    )
    max_output_tokens = 256
    if args.context_pack:
        pack = ContextPackRecord.model_validate_json(open(args.context_pack, encoding="utf-8").read())
        input_text = plan_prompt(pack)
        max_output_tokens = 8192

    response = OpenAI(api_key=api_key, base_url=base_url).responses.create(
        model=model,
        instructions=instructions,
        input=input_text,
        reasoning={"effort": effort},
        max_output_tokens=max_output_tokens,
        timeout=timeout,
    )
    if not getattr(response, "output_text", ""):
        print("Responses API returned no output text", file=sys.stderr)
        return 1
    print("Responses API preflight passed")
    print(f"model={model} base_url={base_url} status={getattr(response, 'status', None)} input_chars={len(input_text)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
