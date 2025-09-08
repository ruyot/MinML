#!/usr/bin/env python3
"""
Compare original vs compressed prompts using OpenAI with real API key.
 - Loads OPENAI_API_KEY from environment (optionally .env via --dotenv)
 - Uses project compression pipeline for prompt compression
 - Counts tokens for original and compressed prompts
 - Sends both prompts to OpenAI and prints responses and usage
"""

import os
import sys
import argparse
import asyncio
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.compression_pipeline import CompressionPipeline
from tokenizer.tokenizer_utils import get_default_tokenizer


def require_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found in environment. Set it or use --dotenv to load .env."
        )
    return api_key


async def openai_chat(prompt: str, model: str, max_tokens: int = 256) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=require_api_key())

    # Use Responses API for gpt-5 with max_completion_tokens; otherwise Chat Completions
    if str(model).startswith("gpt-5"):
        resp = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=max_tokens,
        )
        # Extract text robustly
        text = None
        if hasattr(resp, "output_text") and resp.output_text:
            text = resp.output_text
        elif hasattr(resp, "output") and resp.output:
            try:
                parts = []
                for item in resp.output:
                    if hasattr(item, "content") and item.content:
                        for c in item.content:
                            if hasattr(c, "text") and hasattr(c.text, "value"):
                                parts.append(c.text.value)
                text = "".join(parts) if parts else None
            except Exception:
                text = None
        if not text:
            text = ""

        usage_obj = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage_obj, "input_tokens", None) if usage_obj else None
        completion_tokens = getattr(usage_obj, "output_tokens", None) if usage_obj else None
        total_tokens = getattr(usage_obj, "total_tokens", None) if usage_obj else None
        if total_tokens is None and (prompt_tokens is not None and completion_tokens is not None):
            total_tokens = prompt_tokens + completion_tokens

        return {
            "text": text,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        usage = getattr(resp, "usage", None)
        return {
            "text": resp.choices[0].message.content,
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "total_tokens": usage.total_tokens if usage else None,
            },
        }


async def main():
    parser = argparse.ArgumentParser(description="Compare OpenAI responses with/without compression")
    parser.add_argument("--model", default="gpt-5", help="OpenAI chat model to use (default: gpt-5)")
    parser.add_argument("--prompt", help="Prompt text. If omitted and --interactive, will prompt for input")
    # Force maximum compression; keep arg for compatibility but ignore value
    parser.add_argument("--compression-level", type=int, default=3, choices=[1, 2, 3], help="(Ignored) Compression level; always uses 3")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max completion tokens")
    parser.add_argument("--interactive", action="store_true", help="Read prompt from stdin interactively")
    parser.add_argument("--dotenv", action="store_true", help="Load environment from .env before running")
    args = parser.parse_args()

    if args.dotenv:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception as e:
            raise RuntimeError(f"Failed to load .env: {e}")

    # Enforce key presence now
    require_api_key()

    if args.interactive and not args.prompt:
        try:
            prompt = input("Enter your prompt: ")
        except EOFError:
            prompt = ""
    else:
        prompt = args.prompt or "Explain how a hash map works to a beginner."

    # Build compression pipeline
    tokenizer = get_default_tokenizer()
    pipeline = CompressionPipeline(tokenizer=tokenizer, compression_level=3)

    comp = pipeline.compress_with_stats(prompt)
    compressed = comp["compressed"]
    stats = comp["stats"]

    print("\n=== Prompt Comparison ===")
    print(f"Original:   {len(prompt)} chars, {stats['original_tokens']} tokens")
    print(f"Compressed: {len(compressed)} chars, {stats['compressed_tokens']} tokens")
    print(f"Saved:      {stats['tokens_saved']} tokens ({stats['tokens_saved_percent']:.1f}%)")

    # Query both
    print("\n=== Querying OpenAI (original) ===")
    orig_res = await openai_chat(prompt, model=args.model, max_tokens=args.max_tokens)
    print("\n=== Querying OpenAI (compressed) ===")
    comp_res = await openai_chat(compressed, model=args.model, max_tokens=args.max_tokens)

    # Side-by-side display
    print("\n=== Side-by-Side ===")
    left_title = "Original"
    right_title = "Compressed"
    col_width = 60
    sep = " | "

    def fmt_block(title: str, text: str) -> list:
        lines = [f"[{title}]", ""]
        # Simple wrap
        import textwrap
        lines.extend(textwrap.wrap(text or "", width=col_width))
        return lines

    left = fmt_block("Prompt", prompt)
    right = fmt_block("Prompt", compressed)
    max_len = max(len(left), len(right))
    left += [""] * (max_len - len(left))
    right += [""] * (max_len - len(right))
    print("\nPrompts:")
    for l, r in zip(left, right):
        print(l.ljust(col_width) + sep + r.ljust(col_width))

    left = fmt_block("Response", orig_res["text"]) + ["", f"Usage: {orig_res['usage']}"]
    right = fmt_block("Response", comp_res["text"]) + ["", f"Usage: {comp_res['usage']}"]
    max_len = max(len(left), len(right))
    left += [""] * (max_len - len(left))
    right += [""] * (max_len - len(right))
    print("\nResponses:")
    for l, r in zip(left, right):
        print(l.ljust(col_width) + sep + r.ljust(col_width))

    # Summary
    print("\n=== Summary ===")
    if orig_res["usage"]["total_tokens"] and comp_res["usage"]["total_tokens"]:
        delta = orig_res["usage"]["total_tokens"] - comp_res["usage"]["total_tokens"]
        print(f"Total tokens: original {orig_res['usage']['total_tokens']} vs compressed {comp_res['usage']['total_tokens']} (Δ {delta})")
    else:
        print("Token usage not available from API; showed local token estimates above.")


if __name__ == "__main__":
    asyncio.run(main())


