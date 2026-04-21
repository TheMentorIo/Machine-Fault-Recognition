from __future__ import annotations

import argparse
from pathlib import Path

from audio_preprocessor import AudioPreprocessor, save_wav


def _iter_wavs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    exts = {".wav", ".flac", ".ogg"}
    return sorted(p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess WAV audio (denoise, normalize, trim, pad/crop).")
    parser.add_argument("--input", required=True, help="WAV file or folder containing WAV files")
    parser.add_argument("--output", required=True, help="Output folder for preprocessed WAV files")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--duration", type=float, default=4.0, help="Target duration (seconds)")
    parser.add_argument("--trim-threshold", type=float, default=1e-3, help="Silence trim threshold")
    parser.add_argument("--target-dbfs", type=float, default=-20.0, help="RMS normalization target dBFS")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pre = AudioPreprocessor(
        target_sample_rate=args.sr,
        duration_seconds=args.duration,
        trim_threshold=args.trim_threshold,
        target_dbfs=args.target_dbfs,
    )

    wavs = _iter_wavs(input_path)
    if not wavs:
        raise SystemExit(f"No .wav files found under: {input_path}")

    processed = 0
    for wav_path in wavs:
        audio, sr = pre.preprocess_wav(wav_path)

        # Preserve relative structure under input root when processing a directory.
        if input_path.is_dir():
            rel = wav_path.relative_to(input_path)
            out_path = (output_dir / rel).with_suffix(".wav")
        else:
            out_path = (output_dir / wav_path.name).with_suffix(".wav")

        save_wav(out_path, audio, sr)
        processed += 1

    print(f"Processed {processed} file(s) into: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
