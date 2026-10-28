"""Generate reproducible fictional PDF, DOCX, and MP4 fixtures."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

PROMPT = "Explain your selected case study, describe the central problem, and recommend a practical solution."
CASES = {
    "case_exact_a": (
        "Fictional Lumen Harbor Cooperative reduced food waste by routing surplus produce through neighborhood "
        "cold lockers. Each locker recorded temperature every ten minutes, and a volunteer coordinator matched "
        "nearby requests before the evening pickup window. The pilot measured kilograms rescued, electricity use, "
        "and missed collections for twelve weeks."
    ),
    "case_para_a": (
        "The made-up Brindle Works repair network improved appliance reuse by giving libraries small diagnostic kits. "
        "Residents booked a short inspection, technicians identified replaceable components, and the library lent the "
        "repaired item while its owner waited for a part. The team compared repair completion, waiting time, and repeat "
        "faults across several branches."
    ),
    "case_para_b": (
        "At fictional Brindle Works, tool kits in local libraries helped technicians assess broken household devices. "
        "People reserved an examination, staff found components that could be swapped, and a loaner stayed available "
        "until the replacement arrived. The project tracked successful fixes, delays, and recurring problems at multiple sites."
    ),
    "external_overlap": (
        "A resilient fictional city service keeps a written inventory of critical supplies, checks the list at regular "
        "intervals, and assigns a named person to replenish each item before the reserve falls below its minimum level. "
        "Clear ownership makes routine preparation easier to audit."
    ),
    "original_a": "On the imaginary moon Velora, floating gardens capture mineral mist at dawn. Researchers rotate salt-tolerant seedlings between shaded platforms and measure root growth after each mist cycle.",
    "original_b": "Fictional Emberline Transit uses painted floor markers to help riders find quiet waiting zones. A small community team revises the colors after monthly accessibility walks.",
    "ai_style": (
        "The fictional Ardent Vale observatory carefully records each repeated signal. The consistent process produces "
        "orderly observations. The repeated method supports clear reports. The stable pattern enables predictable summaries. "
        "Each measured event follows the same reliable sequence. The documentation remains regular and structured."
    ),
    "case_crossmodal_a": (
        "The fictional Kestrel Ridge clean energy pilot placed rooftop solar collectors on public transit shelters. "
        "Each shelter stored excess power in a shared neighborhood battery, and a dispatch algorithm rerouted reserve "
        "electricity during peak evening hours. The pilot measured kilowatt-hours generated, battery degradation, "
        "and outage frequency over six months."
    ),
    "benign_c": "Analysis: The invented Cedar Quay ferry should publish hourly occupancy counts so late riders can choose a quieter sailing.",
    "benign_d": "Analysis: Fictional Northlamp clinics could share unused appointment slots through a neighborhood notice board.",
    "benign_a": "Analysis: A small tool library could lower repair costs in the invented town of Mica Bay.",
    "benign_b": "Analysis: Fictional Sunward farms should share irrigation logs during dry months.",
}


def write_docx(path: Path, text: str) -> None:
    paras = "".join(
        f'<w:p><w:r><w:t xml:space="preserve">{escape(line)}</w:t></w:r></w:p>' for line in text.split("\n")
    )
    content = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{paras}<w:sectPr/></w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/></Relationships>',
        )
        archive.writestr("word/document.xml", content)


def write_pdf(path: Path, text: str) -> None:
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PDF generation requires PyMuPDF: pip install PyMuPDF") from exc
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in text.splitlines() or [""]:
        words = line.split()
        current = ""
        if not words:
            y += 24
            continue
        for word in words:
            trial = (current + " " + word).strip()
            if len(trial) > 95:
                page.insert_text((54, y), current)
                y += 16
                current = word
                if y > 760:
                    page = doc.new_page()
                    y = 72
            else:
                current = trial
        if current:
            page.insert_text((54, y), current)
            y += 16
            if y > 760:
                page = doc.new_page()
                y = 72
    doc.save(path)
    doc.close()


def write_silent_mp4(path: Path) -> None:
    try:
        import av
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("MP4 generation requires PyAV: pip install av") from exc
    container = av.open(str(path), mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=2)
    stream.width = 320
    stream.height = 180
    stream.pix_fmt = "yuv420p"
    for i in range(4):
        frame = av.VideoFrame.from_ndarray(
            np.full((180, 320, 3), min(240, 40 + i * 35), dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _speak_wav_windows(wav: Path, text: str) -> bool:
    escaped = text.replace("`", "").replace('"', "'")
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f'$synth.SetOutputToWaveFile("{wav}");'
        f'$synth.Speak("{escaped}");'
        "$synth.Dispose();"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and wav.exists() and wav.stat().st_size > 0


def write_spoken_mp4(path: Path, text: str) -> str:
    """Try TTS + FFmpeg/PyAV. Fall back to a silent video if speech cannot be muxed."""
    ffmpeg = shutil.which("ffmpeg")
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "speech.wav"
        spoken = False
        if espeak:
            run = subprocess.run([espeak, "-w", str(wav), text], capture_output=True, text=True, check=False)
            spoken = run.returncode == 0 and wav.exists()
        if not spoken:
            spoken = _speak_wav_windows(wav, text)
        if spoken and ffmpeg:
            still = Path(tmp) / "still.png"
            subprocess.run(
                [ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=navy:s=320x180:d=1", str(still)],
                capture_output=True,
                check=False,
            )
            mux = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loop",
                    "1",
                    "-i",
                    str(still if still.exists() else wav),
                    "-i",
                    str(wav),
                    "-c:v",
                    "libx264",
                    "-tune",
                    "stillimage",
                    "-c:a",
                    "aac",
                    "-shortest",
                    "-pix_fmt",
                    "yuv420p",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if mux.returncode == 0:
                return "spoken_audio_muxed_with_ffmpeg"
        if spoken:
            try:
                return _mux_wav_with_pyav(path, wav)
            except Exception:
                pass
        write_silent_mp4(path)
        return "silent fallback fixture; spoken words are manifest intent only"


def _mux_wav_with_pyav(path: Path, wav: Path) -> str:
    import av
    import numpy as np

    container = av.open(str(path), mode="w", format="mp4")
    video = container.add_stream("libx264", rate=2)
    video.width = 320
    video.height = 180
    video.pix_fmt = "yuv420p"
    audio = container.add_stream("aac", rate=16000)
    with av.open(str(wav)) as src:
        duration = float(src.duration or 4) / av.time_base if src.duration else 4.0
        frames_needed = max(4, int(duration * 2))
        for i in range(frames_needed):
            frame = av.VideoFrame.from_ndarray(
                np.full((180, 320, 3), 30, dtype=np.uint8), format="rgb24"
            )
            frame.pts = i
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        astream = src.streams.audio[0]
        for aframe in src.decode(astream):
            for packet in audio.encode(aframe):
                container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    container.close()
    return "spoken_audio_muxed_with_pyav"


def generate(root, overwrite: bool = False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    notes = ["All organizations and cases are fictional."]

    def add(name, fmt, category, text, **extra):
        path = root / name
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
        if fmt == "pdf":
            write_pdf(path, text)
        elif fmt == "docx":
            write_docx(path, text)
        elif fmt == "mp4":
            status = extra.get("transcription_status")
            if extra.get("known_spoken_words"):
                extra["transcription_status"] = write_spoken_mp4(path, extra["known_spoken_words"])
            else:
                write_silent_mp4(path)
                extra["transcription_status"] = status or "silent video"
        elif fmt == "empty":
            path.write_bytes(b"")
        entries.append(
            {"document_id": name, "filename": name, "format": fmt, "category": category, "text": text, **extra}
        )

    add("01_exact_original.pdf", "pdf", "exact_copy", PROMPT + "\n\n" + CASES["case_exact_a"])
    add(
        "02_exact_copy.docx",
        "docx",
        "exact_copy",
        PROMPT + "\n\n" + CASES["case_exact_a"],
        known_copied_source="01_exact_original.pdf",
    )
    add("03_paraphrase_a.docx", "docx", "paraphrase", PROMPT + "\n\n" + CASES["case_para_a"])
    add(
        "04_paraphrase_b.docx",
        "docx",
        "paraphrase",
        PROMPT + "\n\n" + CASES["case_para_b"],
        known_paraphrased_relationship="03_paraphrase_a.docx",
    )
    add(
        "05_crossmodal_written.pdf",
        "pdf",
        "cross_modal",
        PROMPT + "\n\n" + CASES["case_crossmodal_a"],
        known_cross_modal_relationship="12_pitch.mp4",
    )
    add("06_reference_overlap.pdf", "pdf", "external_reference", CASES["external_overlap"])
    add("07_original_velora.pdf", "pdf", "boilerplate", PROMPT + "\n\n" + CASES["original_a"], originality="independent_analysis")
    add("08_original_emberline.pdf", "pdf", "boilerplate", PROMPT + "\n\n" + CASES["original_b"], originality="independent_analysis")
    add("09_ai_style.docx", "docx", "ai_style", PROMPT + "\n\n" + CASES["ai_style"])
    add("10_benign_template_a.docx", "docx", "boilerplate", PROMPT + "\n" + CASES["benign_a"])
    add("11_benign_template_b.docx", "docx", "boilerplate", PROMPT + "\n" + CASES["benign_b"])
    add(
        "12_pitch.mp4",
        "mp4",
        "cross_modal",
        "",
        known_spoken_words=CASES["case_crossmodal_a"],
    )
    add("13_silent_pitch.mp4", "mp4", "degraded", "", transcription_status="silent video")
    add("14_corrupt_zero_byte.pdf", "empty", "degraded", "", expected_detection="zero-byte failure")
    add("15_benign_template_c.docx", "docx", "boilerplate", PROMPT + "\n" + CASES["benign_c"])

    audio_note = next((item.get("transcription_status") for item in entries if item["filename"] == "12_pitch.mp4"), "")
    if audio_note and "silent" in str(audio_note):
        notes.append(
            "12_pitch.mp4 is a silent fallback because TTS/FFmpeg muxing was unavailable. "
            "Declared spoken words are manifest intent only and must not be counted as a detected transcript match."
        )
        cross_modal_pairs = []
    else:
        notes.append("12_pitch.mp4 contains generated spoken audio of the known Kestrel Ridge passage.")
        cross_modal_pairs = [["05_crossmodal_written.pdf", "12_pitch.mp4"]]
    notes.append(
        "Five or more written submissions share the assignment prompt and contain independent analysis "
        "(07, 08, 10, 11, 15, and the prompt prefix on several others)."
    )

    manifest = {
        "dataset_version": "1.0",
        "synthetic_only": True,
        "notes": notes,
        "documents": entries,
        "positive_pairs": [
            ["01_exact_original.pdf", "02_exact_copy.docx"],
            ["03_paraphrase_a.docx", "04_paraphrase_b.docx"],
        ],
        "cross_modal_pairs": cross_modal_pairs,
        "negative_benign_pairs": [
            ["10_benign_template_a.docx", "11_benign_template_b.docx"],
            ["10_benign_template_a.docx", "15_benign_template_c.docx"],
            ["11_benign_template_b.docx", "15_benign_template_c.docx"],
            ["07_original_velora.pdf", "08_original_emberline.pdf"],
        ],
    }
    if cross_modal_pairs:
        manifest["positive_pairs"].extend(cross_modal_pairs)
    path = root / "dataset_manifest.json"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./sample_submissions")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    items = generate(args.output_dir, args.overwrite)
    print(f"Generated {len(items)} synthetic submissions in {args.output_dir}")


if __name__ == "__main__":
    main()
