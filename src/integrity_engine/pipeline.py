"""End-to-end deterministic local processing pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from .config import EngineConfig, load_config
from .detection.ai_content import ai_content_indicator
from .detection.boilerplate import BoilerplateDetector
from .detection.external_similarity import find_external_matches, load_references
from .detection.peer_similarity import find_peer_matches
from .detection.scoring import human_review_recommendation, risk_level
from .exceptions import OptionalDependencyError
from .extractors.base import extract
from .preprocessing.chunking import chunk_segments
from .reporting.json_report import write_json
from .reporting.markdown_report import write_markdown
from .reporting.summary import write_summary
from .schemas import Chunk

LOG = logging.getLogger(__name__)
SUPPORTED = {".pdf", ".docx", ".mp4"}
SKIP_PARTS = {".git", "__pycache__", "models", "cache", ".cache", "reports", "tmp", ".test-tmp"}


def discover(input_dir, output_dir=None) -> list[Path]:
    root = Path(input_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Input directory does not exist: {root}")
    excluded = {Path(output_dir).resolve()} if output_dir else set()
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        relative = path.relative_to(root)
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        resolved = path.resolve()
        if any(resolved == folder or folder in resolved.parents for folder in excluded):
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix().casefold())


def _format(path: str) -> str:
    return Path(path).suffix.lower().lstrip(".")


def _is_cross_modal(fmt1: str, fmt2: str) -> bool:
    f1 = fmt1.lower().lstrip(".")
    f2 = fmt2.lower().lstrip(".")
    return (f1 == "mp4" and f2 in {"pdf", "docx"}) or (f2 == "mp4" and f1 in {"pdf", "docx"})


def _pair_scores(query_chunks: list[Chunk], source_chunks: list[Chunk], encoder) -> dict[tuple[str, str], float]:
    from .embeddings.encoder import cosine_matrix

    q_texts = [chunk.normalized_text or " " for chunk in query_chunks]
    s_texts = [chunk.normalized_text or " " for chunk in source_chunks]
    q_vec = encoder.encode(q_texts)
    s_vec = encoder.encode(s_texts)
    matrix = cosine_matrix(q_vec, s_vec)
    scores: dict[tuple[str, str], float] = {}
    for i, query in enumerate(query_chunks):
        for j, source in enumerate(source_chunks):
            scores[(query.chunk_id, source.chunk_id)] = float(matrix[i, j])
    return scores


def _try_encoder(cfg: EngineConfig, offline: bool):
    if offline:
        return None, "Offline mode: lexical overlap matcher used. This is not sentence-transformer cosine similarity."
    try:
        from .embeddings.encoder import SentenceEncoder

        encoder = SentenceEncoder(
            cfg.embedding.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            cfg.embedding.get("device", "cpu"),
            int(cfg.embedding.get("batch_size", 32)),
            cfg.embedding.get("cache_dir"),
        )
        encoder.encode(["health check"])
        return encoder, None
    except OptionalDependencyError as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"Embedding model unavailable ({type(exc).__name__}: {exc}). Lexical matcher used."


def run_pipeline(input_dir, output_dir, config=None, overrides=None):
    started = time.perf_counter()
    cfg = config or load_config()
    overrides = overrides or {}
    offline = bool(overrides.get("offline", cfg.offline))
    root = Path(input_dir).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    files = discover(root, out)
    docs = []
    for path in files:
        doc = extract(
            path,
            root,
            cfg.video.get("whisper_model", "tiny"),
            cfg.video.get("device", "cpu"),
            cfg.video.get("compute_type", "int8"),
            cfg.video.get("language", "en"),
            offline=offline,
        )
        docs.append(doc)
        if doc.error:
            LOG.warning("%s: %s", doc.document_id, doc.error)
        for warning in doc.warnings:
            LOG.info("%s: %s", doc.document_id, warning)

    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(
            chunk_segments(
                doc.document_id,
                doc.path,
                doc.source_format,
                doc.segments,
                cfg.chunking.target_words,
                cfg.chunking.overlap_words,
            )
        )

    detector = BoilerplateDetector(
        int(cfg.boilerplate.get("min_document_frequency", 5)),
        cfg.boilerplate.get("phrases") or [],
        int(cfg.boilerplate.get("min_sentence_words", 5)),
    ).fit(docs)

    encoder, encoder_warning = _try_encoder(cfg, offline)
    semantic = None
    if encoder is not None and chunks:
        try:
            semantic = _pair_scores(chunks, chunks, encoder)
        except Exception as exc:
            encoder_warning = f"Embedding failed during batch encode ({exc}); lexical matcher used."
            encoder = None
            semantic = None

    peer_threshold = overrides.get("peer_threshold", cfg.similarity.peer_semantic_threshold)
    top_k = overrides.get("top_k", cfg.similarity.top_k)
    peer, scores = find_peer_matches(
        chunks, peer_threshold, cfg.similarity.exact_threshold, top_k, detector, semantic
    )

    ref_path = Path(cfg.references)
    if not ref_path.is_absolute():
        ref_path = Path.cwd() / ref_path
    try:
        refs = load_references(ref_path)
    except FileNotFoundError:
        refs = []
        LOG.warning("Reference corpus unavailable: %s", ref_path)

    ext_semantic = None
    ref_chunks: list[Chunk] = []
    if encoder is not None and refs:
        from .preprocessing.chunking import chunk_segments as chunk_refs
        from .schemas import Segment

        for ref in refs:
            ref_chunks.extend(
                chunk_refs(
                    ref["reference_id"],
                    ref.get("title") or ref["reference_id"],
                    "reference",
                    [Segment(ref["text"], {"reference_id": ref["reference_id"]})],
                    cfg.chunking.target_words,
                    0,
                )
            )
        if ref_chunks and chunks:
            try:
                ext_semantic = _pair_scores(chunks, ref_chunks, encoder)
            except Exception as exc:
                LOG.warning("External embedding comparison failed: %s", exc)

    external_threshold = overrides.get("external_threshold", cfg.similarity.external_semantic_threshold)
    ext, ext_scores = find_external_matches(
        chunks, refs, external_threshold, cfg.similarity.exact_threshold, ext_semantic
    )

    max_matches = int(cfg.reports.get("max_matches", 12))
    reports = {}
    for doc in docs:
        peer_matches = [m for m in peer if m["query_document_id"] == doc.document_id][:max_matches]
        ext_matches = [m for m in ext if m["query_document_id"] == doc.document_id][:max_matches]
        multimodal = [
            m
            for m in peer_matches
            if _is_cross_modal(_format(m.get("query_filename") or ""), _format(m.get("matched_file") or ""))
        ]
        text = " ".join(segment.text for segment in doc.segments)
        ai = ai_content_indicator(text, int(cfg.ai_indicator.get("min_words", 100)))
        peer_score = scores.get(doc.document_id, 0.0)
        ext_score = ext_scores.get(doc.document_id, 0.0)
        warnings = list(doc.warnings)
        if doc.error:
            warnings.append(doc.error)
        if encoder_warning:
            warnings.append(encoder_warning)
        if not refs:
            warnings.append("External reference corpus empty or missing; external similarity is a zero-evidence state.")
        if ai["confidence"] == "insufficient":
            warnings.append("AI-content indicator has insufficient text for a stable heuristic reading.")
        evidence = bool(peer_matches or ext_matches)
        risk = risk_level(peer_score, ext_score, ai["score"], evidence, cfg.risk)
        review = human_review_recommendation(doc.status, warnings, peer_score, ext_score, evidence, doc.error)
        scoring_method = "document_coverage_semantic" if semantic is not None else "document_coverage_lexical_fallback"
        reports[doc.document_id] = {
            "schema_version": str(cfg.reports.get("schema_version", "1.0")),
            "document_id": doc.document_id,
            "status": doc.status,
            "total_integrity_risk": risk,
            "scores": {
                "peer_similarity_percentage": peer_score,
                "external_similarity_percentage": ext_score,
                "ai_generated_probability": ai["score"],
            },
            "score_metadata": {
                "ai_score_name": "ai_content_indicator",
                "ai_score_calibrated": False,
                "peer_scoring_method": scoring_method,
                "external_scoring_method": scoring_method if ext_semantic is not None else "document_coverage_lexical_fallback",
                "ai_features": ai["features"],
                "ai_confidence": ai["confidence"],
                "eligible_chunks": sum(chunk.document_id == doc.document_id for chunk in chunks),
                "matching_chunks": len(peer_matches),
            },
            "peer_matches": peer_matches,
            "multimodal_matches": multimodal,
            "external_matches": ext_matches,
            "explanations": [
                "Peer and reference percentages estimate matched eligible query-chunk word coverage after boilerplate discounting.",
                "Values are screening signals, not probabilities of misconduct.",
                "A high AI-content indicator alone is not evidence of plagiarism.",
                review,
            ],
            "human_review_recommendation": review,
            "warnings": warnings,
            "metadata": {
                "source_format": doc.source_format,
                "chunks_processed": sum(chunk.document_id == doc.document_id for chunk in chunks),
                "chunks_matched": len(peer_matches),
                "extraction_error": doc.error,
            },
        }
        safe = doc.document_id.replace("/", "__").replace("\\", "__")
        write_json(reports[doc.document_id], out / f"{safe}.json")
        write_markdown(reports[doc.document_id], out / f"{safe}.md")

    summary = write_summary(reports, docs, out, time.perf_counter() - started)
    LOG.info("Finished %s documents in %.2fs", summary["discovered_documents"], summary["processing_duration_seconds"])
    return docs, reports, summary
