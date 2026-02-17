#!/usr/bin/env python3
"""
Speech-to-Transcript Demo (Additional Challenge)
================================================
Convert clinical audio recordings to transcripts using Whisper.

This is an OPTIONAL component. Pre-generated transcripts are provided,
but teams can experiment with ASR to improve transcript quality.

Usage:
    python speech_to_transcript.py --input evaluation_bundle/RES0198/RES0198.mp3
    python speech_to_transcript.py --input evaluation_bundle/ --output my_transcripts/

Requirements:
    pip install openai-whisper
    # or for faster inference:
    pip install faster-whisper
"""

import argparse
import json
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


def format_transcript(result: dict) -> str:
    """
    Format ASR output as readable transcript.

    Note: Basic Whisper doesn't do speaker diarization (can't distinguish
    Patient vs Doctor). For that, use WhisperX or pyannote-audio.
    """
    lines = []
    for seg in result["segments"]:
        timestamp = f"[{seg['start']:.1f}s - {seg['end']:.1f}s]"
        lines.append(f"{timestamp} {seg['text']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Speech-to-Transcript Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python speech_to_transcript.py --input evaluation_bundle/RES0198/RES0198.mp3
  python speech_to_transcript.py --input evaluation_bundle/ --output asr_output/ --model small
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
    args = parser.parse_args()

    input_path = Path(args.input)

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
            transcript = format_transcript(result)

            if args.output:
                out_dir = Path(args.output)
                out_dir.mkdir(parents=True, exist_ok=True)

                res_id = audio_file.stem

                # Save formatted transcript
                txt_path = out_dir / f"{res_id}_asr.txt"
                with open(txt_path, "w") as f:
                    f.write(f"# {res_id} - ASR Transcript (Whisper {args.model})\n")
                    f.write("# Note: No speaker diarization (P/D labels require WhisperX)\n\n")
                    f.write(transcript)
                print(f"  Saved: {txt_path}")

                # Save JSON with segments
                json_path = out_dir / f"{res_id}_asr.json"
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
