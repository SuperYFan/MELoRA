import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from api_fewshot_baseline import BENCHMARKS, evaluate


DEFAULT_MODEL = "internvl3.5-latest"


def fetch_json(url: str, api_key: str, timeout: int = 120) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps({}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
        return json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {body}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run InternVL API few-shot baselines on Chinese meme benchmarks."
    )
    parser.add_argument("--dataset", choices=sorted(BENCHMARKS))
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--repo_root", type=Path, default=Path("."))
    parser.add_argument("--output_dir", type=Path, default=Path("api_fewshot_results_internvl"))
    parser.add_argument("--eval_split", choices=["auto", "test", "dev", "test_unlabeled"], default="auto")
    parser.add_argument("--num_shots", type=int, default=4)
    parser.add_argument("--shot_mode", choices=["per_class", "total"], default="per_class")
    parser.add_argument(
        "--input_source",
        choices=["image", "original", "generated"],
        default="image",
        help=(
            "image sends few-shot example images and the target image to InternVL. "
            "original/generated use original_input or input as a text-only reference."
        ),
    )
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api_url", required=True, help="OpenAI-compatible chat-completions endpoint.")
    parser.add_argument("--models_url", default="", help="Optional: endpoint that lists the available models.")
    parser.add_argument("--api_key", default="", help="API key for the endpoint given by --api_url.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--output_retries", type=int, default=3)
    parser.add_argument("--retry_base_delay", type=float, default=2.0)
    parser.add_argument("--retry_max_delay", type=float, default=60.0)
    parser.add_argument("--retry_failed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record_failed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument("--label_0_word", default="否")
    parser.add_argument("--label_1_word", default="是")
    parser.add_argument("--use_image", action="store_true")
    parser.add_argument("--max_image_side", type=int, default=1024)
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument(
        "--json_mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Some endpoints do not support OpenAI JSON mode; it is disabled by default.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--list_models", action="store_true")
    args = parser.parse_args()

    args.api_key = (args.api_key or "").strip()
    if not args.api_key:
        raise EnvironmentError("Please pass --api_key.")
    if args.eval_split == "auto" and args.dataset:
        args.eval_split = BENCHMARKS[args.dataset]["default_eval_split"]
    args.data_root = args.data_root.resolve()
    args.repo_root = args.repo_root.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


def main() -> None:
    args = parse_args()
    if args.list_models:
        response = fetch_json(args.models_url, args.api_key, timeout=args.timeout)
        print(json.dumps(response, ensure_ascii=False, indent=2))
        data = response.get("data")
        if data is not None:
            print(json.dumps(data, ensure_ascii=False, indent=2))
        if not args.dataset:
            return
    if not args.dataset:
        raise ValueError("Please provide --dataset, or use --list_models to only query available models.")
    evaluate(args)


if __name__ == "__main__":
    main()
