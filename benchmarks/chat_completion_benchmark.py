#!/usr/bin/env python3

import argparse
import json
import socket
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request


DEFAULT_API_BASE = "http://127.0.0.1:8000"
DEFAULT_MODEL = "gemma-4-26b-a4b-it-nvfp4a16"
DEFAULT_OUTPUT = "benchmarks/gemma4-26b-a4b-it-nvfp4a16-metrics.md"


@dataclass
class BenchmarkCase:
    name: str
    prompt_kind: str
    prompt_content: str
    max_tokens: int
    run_index: int


@dataclass
class BenchmarkResult:
    name: str
    prompt_kind: str
    run_index: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    ttft_seconds: float
    total_latency_seconds: float
    prefill_tps: float
    decode_tps: float
    output_tps: float
    total_tps: float


def post_json(url: str, payload: dict, timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def stream_chat_completion(
    api_base: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    timeout: int,
) -> tuple[dict, float, float]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{api_base}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    usage = None
    start = time.perf_counter()
    ttft = None

    with request.urlopen(req, timeout=timeout) as resp:
        while True:
            raw_line = resp.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue

            payload_text = line[6:]
            if payload_text == "[DONE]":
                break

            chunk = json.loads(payload_text)
            if chunk.get("usage"):
                usage = chunk["usage"]

            for choice in chunk.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content and ttft is None:
                    ttft = time.perf_counter() - start

    total_latency = time.perf_counter() - start
    if usage is None:
        raise RuntimeError("Streaming response ended without a usage block")
    if ttft is None:
        ttft = total_latency
    return usage, ttft, total_latency


def tokenize_count(api_base: str, model: str, prompt: str, timeout: int) -> int:
    response = post_json(
        f"{api_base}/tokenize",
        {"model": model, "prompt": prompt},
        timeout=timeout,
    )
    return int(response["count"])


def build_small_prompt(run_index: int) -> str:
    return (
        f"Benchmark run nonce small-{run_index:02d}. "
        "Respond by repeating the word SMALL_BENCH separated by spaces until you hit the token limit."
    )


def build_large_prompt(api_base: str, model: str, target_tokens: int, run_index: int, timeout: int) -> str:
    prefix = (
        f"Benchmark run nonce large-{run_index:02d}. "
        "Read the following context and then respond by repeating the word LARGE_BENCH separated by spaces until you hit the token limit.\n"
    )
    suffix = "\nEnd of benchmark context."
    unit = "benchmark "

    low = 1
    high = 1
    while True:
        candidate = prefix + (unit * high) + suffix
        if tokenize_count(api_base, model, candidate, timeout) >= target_tokens:
            break
        high *= 2

    while low < high:
        mid = (low + high + 1) // 2
        candidate = prefix + (unit * mid) + suffix
        count = tokenize_count(api_base, model, candidate, timeout)
        if count <= target_tokens:
            low = mid
        else:
            high = mid - 1

    return prefix + (unit * low) + suffix


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def run_case(api_base: str, model: str, case: BenchmarkCase, timeout: int) -> BenchmarkResult:
    messages = [{"role": "user", "content": case.prompt_content}]
    usage, ttft, total_latency = stream_chat_completion(
        api_base=api_base,
        model=model,
        messages=messages,
        max_tokens=case.max_tokens,
        timeout=timeout,
    )

    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    total_tokens = int(usage["total_tokens"])
    decode_window = max(total_latency - ttft, 1e-9)

    return BenchmarkResult(
        name=case.name,
        prompt_kind=case.prompt_kind,
        run_index=case.run_index,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        ttft_seconds=ttft,
        total_latency_seconds=total_latency,
        prefill_tps=safe_div(prompt_tokens, ttft),
        decode_tps=safe_div(completion_tokens, decode_window),
        output_tps=safe_div(completion_tokens, total_latency),
        total_tps=safe_div(total_tokens, total_latency),
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def fmt(value: float) -> str:
    return f"{value:,.2f}"


def build_report(
    api_base: str,
    model: str,
    max_tokens: int,
    large_target_tokens: int,
    results: list[BenchmarkResult],
) -> str:
    grouped: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result.name, []).append(result)

    summary_rows = []
    for name, items in grouped.items():
        summary_rows.append(
            [
                name,
                str(len(items)),
                fmt(statistics.mean(item.prompt_tokens for item in items)),
                fmt(statistics.mean(item.completion_tokens for item in items)),
                fmt(statistics.mean(item.ttft_seconds for item in items)),
                fmt(statistics.mean(item.total_latency_seconds for item in items)),
                fmt(statistics.mean(item.prefill_tps for item in items)),
                fmt(statistics.mean(item.decode_tps for item in items)),
                fmt(statistics.mean(item.output_tps for item in items)),
                fmt(statistics.mean(item.total_tps for item in items)),
            ]
        )

    detail_rows = []
    for result in results:
        detail_rows.append(
            [
                result.name,
                str(result.run_index),
                str(result.prompt_tokens),
                str(result.completion_tokens),
                str(result.total_tokens),
                fmt(result.ttft_seconds),
                fmt(result.total_latency_seconds),
                fmt(result.prefill_tps),
                fmt(result.decode_tps),
                fmt(result.output_tps),
                fmt(result.total_tps),
            ]
        )

    lines = [
        "# Gemma4 NVFP4 Chat Completion Benchmark",
        "",
        f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- Host: `{socket.gethostname()}`",
        f"- API base: `{api_base}`",
        f"- Model: `{model}`",
        f"- Max completion tokens per request: `{max_tokens}`",
        f"- Large prompt raw token target: `{large_target_tokens}`",
        "- Method: streamed chat completions with `stream_options.include_usage=true`",
        "- Cache note: each run starts with a unique nonce to avoid prompt-prefix cache reuse",
        "",
        "## Summary",
        "",
        markdown_table(
            [
                "Case",
                "Runs",
                "Avg prompt toks",
                "Avg completion toks",
                "Avg TTFT s",
                "Avg total s",
                "Avg prefill tok/s",
                "Avg decode tok/s",
                "Avg output tok/s",
                "Avg total tok/s",
            ],
            summary_rows,
        ),
        "",
        "## Detailed Runs",
        "",
        markdown_table(
            [
                "Case",
                "Run",
                "Prompt toks",
                "Completion toks",
                "Total toks",
                "TTFT s",
                "Total s",
                "Prefill tok/s",
                "Decode tok/s",
                "Output tok/s",
                "Total tok/s",
            ],
            detail_rows,
        ),
        "",
        "## Notes",
        "",
        "- `Prefill tok/s` is approximated as `prompt_tokens / TTFT`.",
        "- `Decode tok/s` is approximated as `completion_tokens / (total_latency - TTFT)`.",
        "- `Output tok/s` is `completion_tokens / total_latency`.",
        "- `Total tok/s` is `(prompt_tokens + completion_tokens) / total_latency`.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark streamed chat completion throughput")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--large-target-tokens", type=int, default=240000)
    parser.add_argument("--timeout", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cases: list[BenchmarkCase] = []
    for run_index in range(1, args.runs + 1):
        cases.append(
            BenchmarkCase(
                name="small-prompt",
                prompt_kind="small",
                prompt_content=build_small_prompt(run_index),
                max_tokens=args.max_tokens,
                run_index=run_index,
            )
        )
        cases.append(
            BenchmarkCase(
                name="large-prompt",
                prompt_kind="large",
                prompt_content=build_large_prompt(
                    api_base=args.api_base,
                    model=args.model,
                    target_tokens=args.large_target_tokens,
                    run_index=run_index,
                    timeout=args.timeout,
                ),
                max_tokens=args.max_tokens,
                run_index=run_index,
            )
        )

    results: list[BenchmarkResult] = []
    for case in cases:
        print(f"Running {case.name} run {case.run_index}...", flush=True)
        results.append(
            run_case(
                api_base=args.api_base,
                model=args.model,
                case=case,
                timeout=args.timeout,
            )
        )

    report = build_report(
        api_base=args.api_base,
        model=args.model,
        max_tokens=args.max_tokens,
        large_target_tokens=args.large_target_tokens,
        results=results,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Wrote report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
