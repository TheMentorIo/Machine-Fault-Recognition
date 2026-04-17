from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from ..utils.logging import info


def infer_audio_label(path: Path, root: Path) -> str:
    relative_parts = path.relative_to(root).parts
    if len(relative_parts) >= 3:
        machine = relative_parts[-3]
        state = relative_parts[-2]
        return f"{machine}__{state}"
    if len(relative_parts) >= 2:
        return relative_parts[-2]
    return path.parent.name


@dataclass
class DatasetScanner:
    root: Path

    def scan(self) -> tuple[pd.DataFrame, str]:
        wav_files = sorted([path for path in self.root.rglob("*.wav") if path.is_file()])
        csv_files = sorted([path for path in self.root.rglob("*.csv") if path.is_file()])
        info(f"Found {len(wav_files)} WAV files | {len(csv_files)} CSV files")

        if wav_files:
            rows = []
            for path in tqdm(wav_files, desc="Scanning WAV files", unit="file"):
                rows.append(
                    {
                        "path": str(path),
                        "label": infer_audio_label(path, self.root),
                        "sample_id": path.stem,
                    }
                )
            frame = pd.DataFrame(rows)
            return frame, "audio"

        if csv_files:
            frames = [pd.read_csv(path) for path in csv_files]
            frame = pd.concat(frames, ignore_index=True)
            label_column = next((column for column in frame.columns if column.lower() in {"label", "target", "class", "fault", "y"}), None)
            if label_column is None:
                raise RuntimeError("Could not find a label column in CSV data.")
            frame = frame.rename(columns={label_column: "label"})
            return frame, "tabular"

        raise RuntimeError(f"No audio or CSV files found under {self.root}")
