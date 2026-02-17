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

    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language="en")

    # Format as turns (basic speaker diarization placeholder)
    segments = []
    for i, seg in enumerate(result["segments"]):
        segments.append({
            "turn_id": i + 1,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        })

    return {
        "audio_file": audio_path,
        "full_text": result["text"],
        "segments": segments
    }


def transcribe_faster_whisper(audio_path: str, model_size: str = "base") -> dict:
    """Transcribe using Faster-Whisper (4x faster, lower memory)."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    segments, info = model.transcribe(audio_path, language="en")

    result_segments = []
    full_text_parts = []
    for i, seg in enumerate(segments):
        result_segments.append({
            "turn_id": i + 1,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip()
        })
        full_text_parts.append(seg.text.strip())

    return {
        "audio_file": audio_path,
        "full_text": " ".join(full_text_parts),
        "segments": result_segments
    }


def format_as_clinical_transcript(result: dict, res_id: str) -> str:
    """
    Format ASR output as clinical transcript with turn markers.

    Note: Real clinical transcripts have P-X (Patient) and D-X (Doctor) markers.
    Basic ASR doesn't do speaker diarization. For proper speaker separation,
    consider using:
    - WhisperX (https://github.com/m-bain/whisperX)
    - pyannote-audio for diarization
    """
    lines = [f"# {res_id} - ASR Generated Transcript\n"]
    lines.append("# Note: Speaker labels (P/D) require diarization model\n\n")

    for seg in result["segments"]:
        # Without diarization, we can't distinguish P vs D
        # Use generic [T-X] format
        lines.append(f"[T-{seg['turn_id']}] {seg['text']}\n")

    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Speech-to-Transcript Demo")
    parser.add_argument("--input", type=str, required=True,
                       help="Audio file or directory containing audio files")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for transcripts")
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
        audio_files = list(input_path.glob("**/*.mp3"))

    if not audio_files:
        print(f"No audio files found in {input_path}")
        return

    print(f"Found {len(audio_files)} audio file(s)")
    print(f"Using {args.backend} with model size: {args.model}")

    # Select transcription function
    if args.backend == "faster-whisper":
        transcribe_fn = transcribe_faster_whisper
    else:
        transcribe_fn = transcribe_whisper

    # Process each file
    for audio_file in audio_files:
        print(f"\nProcessing: {audio_file.name}")

        try:
            result = transcribe_fn(str(audio_file), args.model)

            # Extract res_id from filename
            res_id = audio_file.stem

            # Format output
            transcript = format_as_clinical_transcript(result, res_id)

            # Save or print
            if args.output:
                out_dir = Path(args.output)
                out_dir.mkdir(parents=True, exist_ok=True)

                # Save formatted transcript
                txt_path = out_dir / f"{res_id}_asr.txt"
                with open(txt_path, "w") as f:
                    f.write(transcript)

                # Save raw JSON
                json_path = out_dir / f"{res_id}_asr.json"
                with open(json_path, "w") as f:
                    json.dump(result, f, indent=2)

                print(f"  Saved: {txt_path}")
            else:
                print(transcript[:500] + "..." if len(transcript) > 500 else transcript)

        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
