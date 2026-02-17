#!/usr/bin/env python3
"""
Speech-to-Transcript Demo (Additional Challenge)
================================================
Convert clinical audio recordings to transcripts using Whisper.

This is an OPTIONAL component. Pre-generated transcripts are provided,
but teams can experiment with ASR to improve transcript quality.

Usage:
    # Basic transcription
    python speech_to_transcript.py --input evaluation_bundle/RES0198/RES0198.mp3

    # With rule-based speaker labels (D/P alternating)
    python speech_to_transcript.py --input evaluation_bundle/RES0198/RES0198.mp3 --add-speakers

    # Evaluate ASR accuracy against ground truth
    python speech_to_transcript.py --evaluate --input evaluation_bundle/

Requirements:
    pip install openai-whisper jiwer
    conda install ffmpeg
"""

import argparse
import json
import re
from pathlib import Path


def transcribe_whisper(audio_path: str, model_size: str = "base") -> dict:
    """Transcribe using OpenAI Whisper."""
    import whisper

    print(f"  Loading whisper model: {model_size}...")
    model = whisper.load_model(model_size)

    print(f"  Transcribing...")
    result = model.transcribe(audio_path, language="en")

    segments = []
    for i, seg in enumerate(result["segments"]):
        segments.append({
            "id": i + 1,
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip()
        })

    return {
        "audio_file": str(audio_path),
        "full_text": result["text"],
        "segments": segments
    }


def transcribe_faster_whisper(audio_path: str, model_size: str = "base") -> dict:
    """Transcribe using Faster-Whisper (4x faster, lower memory)."""
    from faster_whisper import WhisperModel

    print(f"  Loading faster-whisper model: {model_size}...")
    model = WhisperModel(model_size, device="auto", compute_type="auto")

    print(f"  Transcribing...")
    segments_gen, info = model.transcribe(audio_path, language="en")

    segments = []
    full_text_parts = []
    for i, seg in enumerate(segments_gen):
        segments.append({
            "id": i + 1,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip()
        })
        full_text_parts.append(seg.text.strip())

    return {
        "audio_file": str(audio_path),
        "full_text": " ".join(full_text_parts),
        "segments": segments
    }


def add_speaker_labels(segments: list) -> list:
    """
    Add rule-based speaker labels (D/P alternating).

    Assumption: Doctor speaks first, then alternates with Patient.
    This is a simple heuristic - for real diarization use WhisperX/pyannote.
    """
    labeled = []
    # Doctor starts first
    current_speaker = "D"
    d_count, p_count = 0, 0

    for seg in segments:
        if current_speaker == "D":
            d_count += 1
            label = f"D-{d_count}"
            next_speaker = "P"
        else:
            p_count += 1
            label = f"P-{p_count}"
            next_speaker = "D"

        labeled.append({
            **seg,
            "speaker": current_speaker,
            "turn_id": label
        })
        current_speaker = next_speaker

    return labeled


def format_transcript(segments: list, with_speakers: bool = False) -> str:
    """Format segments as readable transcript."""
    lines = []
    for seg in segments:
        if with_speakers and "turn_id" in seg:
            prefix = f"[{seg['turn_id']}]"
        else:
            prefix = f"[{seg['start']:.1f}s - {seg['end']:.1f}s]"
        lines.append(f"{prefix} {seg['text']}")
    return "\n".join(lines)


def load_ground_truth(txt_path: Path) -> str:
    """Load and normalize ground truth transcript."""
    with open(txt_path) as f:
        content = f.read()

    # Remove speaker labels [D-1], [P-2], etc. and "D:" "P:" prefixes
    content = re.sub(r'\[?[DP]-?\d+\]?\s*[DP]?:?\s*', '', content)
    # Remove extra whitespace
    content = ' '.join(content.split())
    return content.lower()


def normalize_text(text: str) -> str:
    """Normalize text for WER comparison."""
    # Remove punctuation, lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    return ' '.join(text.split())


def calculate_wer(reference: str, hypothesis: str) -> dict:
    """Calculate Word Error Rate and other metrics."""
    from jiwer import wer, mer, wil

    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    return {
        "wer": round(wer(ref_norm, hyp_norm) * 100, 2),  # Word Error Rate (%)
        "mer": round(mer(ref_norm, hyp_norm) * 100, 2),  # Match Error Rate (%)
        "wil": round(wil(ref_norm, hyp_norm) * 100, 2),  # Word Information Lost (%)
        "ref_words": len(ref_norm.split()),
        "hyp_words": len(hyp_norm.split()),
    }


def evaluate_all(input_dir: Path, model_size: str, backend: str):
    """Evaluate ASR on all audio files against ground truth."""
    import whisper

    print(f"Loading whisper model: {model_size}...")
    model = whisper.load_model(model_size)

    # Find all patient folders with both mp3 and txt
    results = []
    audio_files = sorted(input_dir.glob("**/*.mp3"))

    print(f"Found {len(audio_files)} audio files\n")
    print(f"{'Patient':<10} {'WER%':<8} {'MER%':<8} {'Words(ref/hyp)':<15}")
    print("-" * 45)

    for audio_file in audio_files:
        res_id = audio_file.stem
        txt_file = audio_file.parent / f"{res_id}.txt"

        if not txt_file.exists():
            print(f"{res_id:<10} [no ground truth]")
            continue

        try:
            # Transcribe
            result = model.transcribe(str(audio_file), language="en")
            asr_text = result["text"]

            # Load ground truth
            gt_text = load_ground_truth(txt_file)

            # Calculate metrics
            metrics = calculate_wer(gt_text, asr_text)
            metrics["res_id"] = res_id
            results.append(metrics)

            print(f"{res_id:<10} {metrics['wer']:<8.1f} {metrics['mer']:<8.1f} {metrics['ref_words']}/{metrics['hyp_words']}")

        except Exception as e:
            print(f"{res_id:<10} [error: {e}]")

    # Summary
    if results:
        avg_wer = sum(r["wer"] for r in results) / len(results)
        avg_mer = sum(r["mer"] for r in results) / len(results)
        print("-" * 45)
        print(f"{'AVERAGE':<10} {avg_wer:<8.1f} {avg_mer:<8.1f}")
        print(f"\nAccuracy: {100 - avg_wer:.1f}% (1 - WER)")
        print(f"Evaluated {len(results)} files with Whisper {model_size}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Speech-to-Transcript Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription
  python speech_to_transcript.py --input evaluation_bundle/RES0198/RES0198.mp3

  # With speaker labels
  python speech_to_transcript.py --input evaluation_bundle/RES0198/RES0198.mp3 --add-speakers

  # Evaluate accuracy on all files
  python speech_to_transcript.py --evaluate --input evaluation_bundle/ --model base
        """
    )
    parser.add_argument("--input", type=str, required=True,
                       help="Audio file (.mp3) or directory")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (if not set, prints to stdout)")
    parser.add_argument("--model", type=str, default="base",
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size (default: base)")
    parser.add_argument("--backend", type=str, default="whisper",
                       choices=["whisper", "faster-whisper"],
                       help="ASR backend (default: whisper)")
    parser.add_argument("--add-speakers", action="store_true",
                       help="Add rule-based speaker labels (D/P alternating)")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate ASR accuracy against ground truth transcripts")
    args = parser.parse_args()

    input_path = Path(args.input)

    # Evaluate mode
    if args.evaluate:
        if not input_path.is_dir():
            print("--evaluate requires a directory with audio + transcript files")
            return
        evaluate_all(input_path, args.model, args.backend)
        return

    # Find audio files
    if input_path.is_file():
        audio_files = [input_path]
    else:
        audio_files = sorted(input_path.glob("**/*.mp3"))

    if not audio_files:
        print(f"No .mp3 files found in {input_path}")
        return

    print(f"Found {len(audio_files)} audio file(s)")
    print(f"Backend: {args.backend}, Model: {args.model}\n")

    # Select backend
    if args.backend == "faster-whisper":
        transcribe_fn = transcribe_faster_whisper
    else:
        transcribe_fn = transcribe_whisper

    for audio_file in audio_files:
        print(f"Processing: {audio_file.name}")

        try:
            result = transcribe_fn(str(audio_file), args.model)
            segments = result["segments"]

            # Add speaker labels if requested
            if args.add_speakers:
                segments = add_speaker_labels(segments)

            transcript = format_transcript(segments, with_speakers=args.add_speakers)

            if args.output:
                out_dir = Path(args.output)
                out_dir.mkdir(parents=True, exist_ok=True)

                res_id = audio_file.stem
                suffix = "_asr_speakers" if args.add_speakers else "_asr"

                # Save formatted transcript
                txt_path = out_dir / f"{res_id}{suffix}.txt"
                with open(txt_path, "w") as f:
                    f.write(f"# {res_id} - ASR Transcript (Whisper {args.model})\n")
                    if args.add_speakers:
                        f.write("# Speaker labels: Rule-based D/P alternating (not true diarization)\n\n")
                    else:
                        f.write("# Note: No speaker diarization (use --add-speakers for rule-based labels)\n\n")
                    f.write(transcript)
                print(f"  Saved: {txt_path}")

                # Save JSON with segments
                json_path = out_dir / f"{res_id}{suffix}.json"
                result["segments"] = segments
                with open(json_path, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"  Saved: {json_path}")
            else:
                print(f"\n{transcript}\n")

        except Exception as e:
            print(f"  Error: {e}")

        print()


if __name__ == "__main__":
    main()
