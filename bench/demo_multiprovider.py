"""Multi-provider demo: one gate key, two providers, the Anthropic SDK.

The client holds only a gateway key and points its base URL at the gate. It
calls a Claude model and a DeepSeek model, streaming and non-streaming; only
the ``model`` field differs. Prompts are synthetic.

    GATE_KEY=gk_... uv run --with anthropic python -m bench.demo_multiprovider \
        --models claude-sonnet-5-5 deepseek-flash

Then: ``api-gateway report --by provider``.
"""

import argparse
import os
import sys

PROMPT = "Reply with exactly three words describing a lighthouse."


def main() -> int:
    try:
        import anthropic
    except ImportError:
        print(
            "install the SDK: uv run --with anthropic python -m bench.demo_multiprovider",
            file=sys.stderr,
        )
        return 2

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("GATE_URL", "http://localhost:8787"))
    parser.add_argument("--models", nargs="+", default=["claude-sonnet-5-5", "deepseek-flash"])
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    key = os.environ.get("GATE_KEY")
    if not key or not key.startswith("gk_"):
        print(
            "set GATE_KEY to a gateway key (gk_...); provider keys stay in the gate",
            file=sys.stderr,
        )
        return 2
    client = anthropic.Anthropic(api_key=key, base_url=args.base_url, max_retries=0)

    failures = 0
    for model in args.models:
        messages = [{"role": "user", "content": PROMPT}]
        try:
            reply = client.messages.create(
                model=model, max_tokens=args.max_tokens, messages=messages
            )
            text = "".join(b.text for b in reply.content if b.type == "text")
            print(f"[{model}] non-stream: {text!r} usage={reply.usage.model_dump()}")

            with client.messages.stream(
                model=model, max_tokens=args.max_tokens, messages=messages
            ) as stream:
                streamed = "".join(stream.text_stream)
                final = stream.get_final_message()
            print(f"[{model}] stream:     {streamed!r} usage={final.usage.model_dump()}")
        except anthropic.APIStatusError as exc:
            failures += 1
            gateway_code = exc.response.headers.get("x-api-gateway-error", "upstream")
            print(f"[{model}] HTTP {exc.status_code} ({gateway_code}): {exc.message}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
