"""CLI entry point for the integrity engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from integrity_engine.config import load_config
from integrity_engine.logging_config import configure_logging
from integrity_engine.pipeline import run_pipeline

WHISPER_SIZES = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local multimodal submission integrity analysis on PDF, DOCX, and MP4 files."
    )
    parser.add_argument("--input_dir", default="./sample_submissions", help="Directory of submissions")
    parser.add_argument("--output_dir", default="./reports", help="Report output directory")
    parser.add_argument("--config", help="Path to YAML config (defaults to config/default.yaml)")
    parser.add_argument("--model", help="Embedding model name, or a Whisper size (tiny/base/...)")
    parser.add_argument("--device", help="cpu or cuda for embeddings and transcription")
    parser.add_argument("--batch_size", type=int, help="Embedding batch size")
    parser.add_argument("--peer_threshold", type=float, help="Peer cosine/lexical threshold in [0,1]")
    parser.add_argument("--external_threshold", type=float, help="External reference threshold in [0,1]")
    parser.add_argument("--top_k", type=int, help="Maximum sources retained per query chunk")
    parser.add_argument("--offline", action="store_true", help="Skip model downloads; use lexical matching")
    parser.add_argument("--log_level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        cfg = load_config(args.config)
        if args.model:
            if args.model in WHISPER_SIZES:
                cfg.video["whisper_model"] = args.model
            else:
                cfg.embedding["model"] = args.model
        if args.device:
            cfg.video["device"] = args.device
            cfg.embedding["device"] = args.device
        if args.batch_size:
            cfg.embedding["batch_size"] = args.batch_size
        if args.offline:
            cfg.offline = True
        overrides = {
            key: value
            for key, value in {
                "peer_threshold": args.peer_threshold,
                "external_threshold": args.external_threshold,
                "top_k": args.top_k,
                "offline": args.offline,
            }.items()
            if value is not None and value is not False
        }
        if args.offline:
            overrides["offline"] = True
        docs, _reports, summary = run_pipeline(args.input_dir, args.output_dir, cfg, overrides)
        print(
            f"Processed {summary['discovered_documents']} documents; "
            f"{summary['failed']} failed, {summary['empty_or_unextractable']} empty."
        )
        if summary["discovered_documents"] == 0:
            print("No supported PDF/DOCX/MP4 files were found.")
            return 1
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
