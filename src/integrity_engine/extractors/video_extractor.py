"""Video validation, audio extraction, and lazy faster-whisper transcription."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from ..exceptions import OptionalDependencyError
from ..schemas import Segment

LOG = logging.getLogger(__name__)
_MODEL_CACHE: dict[tuple[str, str, str], object] = {}


def extract_video(
    path: Path,
    model_name: str = "tiny",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "en",
    offline: bool = False,
) -> tuple[list[Segment], list[str]]:
    warnings: list[str] = []
    _validate_container(path)

    if not _has_audio_stream(path):
        return [], ["Video has no audio stream; inspected successfully without a transcript."]

    if offline:
        return [], ["--offline skipped speech recognition. Audio was not transcribed."]

    wav_needed = True
    with tempfile.TemporaryDirectory(prefix="integrity-audio-") as tmp:
        wav = Path(tmp) / "audio.wav"
        extract_warnings = _extract_wav(path, wav)
        warnings.extend(extract_warnings)
        if not wav.exists() or wav.stat().st_size == 0:
            raise ValueError("Audio extraction produced an empty WAV file")
        if _wav_is_silent(wav):
            warnings.append("Audio track appears silent; transcript is empty and was not fabricated.")
            return [], warnings
        segments = _transcribe(wav, model_name, device, compute_type, language)
        wav_needed = False  # noqa: F841 - tempfile cleanup always runs
    if not segments:
        warnings.append("Speech recognition returned no meaningful text; video was inspected successfully.")
    return segments, warnings


def _validate_container(path: Path) -> None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode:
            raise ValueError(f"ffprobe rejected video: {probe.stderr.strip() or probe.stdout.strip()}")
        return
    try:
        import av
    except ImportError as exc:
        raise OptionalDependencyError(
            "MP4 validation requires ffprobe on PATH, or PyAV (pip install av) as a documented fallback."
        ) from exc
    try:
        with av.open(str(path)) as container:
            if not container.streams:
                raise ValueError("PyAV opened the file but found no media streams")
    except Exception as exc:
        raise ValueError(f"Unreadable MP4: {exc}") from exc


def _has_audio_stream(path: Path) -> bool:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-show_streams", "-of", "json", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode:
            raise ValueError(f"ffprobe rejected video: {probe.stderr.strip()}")
        streams = json.loads(probe.stdout or "{}").get("streams", [])
        return any(stream.get("codec_type") == "audio" for stream in streams)
    import av

    with av.open(str(path)) as container:
        return any(s.type == "audio" for s in container.streams)


def _extract_wav(path: Path, wav: Path) -> list[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        run = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                str(wav),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if run.returncode:
            raise ValueError(f"FFmpeg audio extraction failed: {run.stderr.strip()}")
        return []
    try:
        import av
        import numpy as np
    except ImportError as exc:
        raise OptionalDependencyError(
            "MP4 transcription requires FFmpeg on PATH. Optional fallback: pip install av. "
            "Install FFmpeg from https://ffmpeg.org and confirm `ffmpeg -version`."
        ) from exc
    LOG.warning("FFmpeg missing; extracting audio with PyAV fallback for %s", path.name)
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("No audio stream")
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
        frames: list[bytes] = []
        for audio_frame in container.decode(stream):
            resampled = resampler.resample(audio_frame)
            if resampled is None:
                continue
            pieces = resampled if isinstance(resampled, list) else [resampled]
            for piece in pieces:
                arr = piece.to_ndarray()
                frames.append(np.asarray(arr, dtype=np.int16).tobytes())
        flushed = resampler.resample(None)
        if flushed:
            pieces = flushed if isinstance(flushed, list) else [flushed]
            for piece in pieces:
                frames.append(np.asarray(piece.to_ndarray(), dtype=np.int16).tobytes())
        payload = b"".join(frames)
    with wave.open(str(wav), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(payload)
    return ["FFmpeg was not on PATH; audio was extracted with the PyAV fallback."]


def _wav_is_silent(path: Path, rms_threshold: float = 80.0) -> bool:
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        width = handle.getsampwidth()
    if not frames:
        return True
    import numpy as np

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width, np.int16)
    samples = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if samples.size == 0:
        return True
    return float(np.sqrt(np.mean(samples**2))) < rms_threshold


def _transcribe(
    wav: Path,
    model_name: str,
    device: str,
    compute_type: str,
    language: str,
) -> list[Segment]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise OptionalDependencyError(
            "Transcription requires faster-whisper: pip install faster-whisper. "
            "Then allow a one-time download of the configured Whisper model (default: tiny)."
        ) from exc
    key = (model_name, device, compute_type)
    if key not in _MODEL_CACHE:
        try:
            _MODEL_CACHE[key] = WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception as exc:
            raise OptionalDependencyError(
                f"Whisper model '{model_name}' could not be loaded: {exc}. "
                "Install faster-whisper and allow the model download, or set video.whisper_model in config."
            ) from exc
    model = _MODEL_CACHE[key]
    segments, _info = model.transcribe(str(wav), language=language)
    out: list[Segment] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        confidence = None
        if getattr(seg, "avg_logprob", None) is not None:
            confidence = {"avg_logprob": float(seg.avg_logprob)}
        location = {
            "start_seconds": float(seg.start),
            "end_seconds": float(seg.end),
        }
        if confidence:
            location["confidence"] = confidence
        out.append(
            Segment(
                text=text,
                location=location,
                start=float(seg.start),
                end=float(seg.end),
                original_text=text,
            )
        )
    return out
