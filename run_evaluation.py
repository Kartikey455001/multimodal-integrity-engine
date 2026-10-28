"""Evaluate pairwise detection against the synthetic manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from integrity_engine.evaluation import evaluate_pairwise
from integrity_engine.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run detection then pairwise evaluation against ground truth.")
    parser.add_argument("--input_dir", default="./sample_submissions")
    parser.add_argument("--output_dir", default="./reports")
    parser.add_argument("--manifest", default="./sample_submissions/dataset_manifest.json")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    docs, reports, _summary = run_pipeline(args.input_dir, args.output_dir, overrides={"offline": args.offline} if args.offline else None)
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    result = evaluate_pairwise(docs, reports, json.loads(manifest_path.read_text(encoding="utf-8")))
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    (Path(args.output_dir) / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
