from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

try:  # Optional: supports FLAC/OGG/etc. without pulling in librosa/numba.
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover
    sf = None


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Convert common PCM formats into mono float32 in [-1, 1] (best-effort)."""
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32_768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2_147_483_648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    else:
        audio = audio.astype(np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def load_wav(path: str | Path, target_sample_rate: int) -> tuple[np.ndarray, int]:
    """Load a WAV file and resample to target_sample_rate if needed."""
    path = Path(path)
    sample_rate, audio = wavfile.read(path)
    audio = _to_mono_float32(audio)
    if int(sample_rate) != int(target_sample_rate):
        gcd = int(np.gcd(int(sample_rate), int(target_sample_rate)))
        audio = signal.resample_poly(audio, int(target_sample_rate) // gcd, int(sample_rate) // gcd).astype(np.float32)
        sample_rate = int(target_sample_rate)
    return audio, int(sample_rate)


def load_audio_any(path: str | Path, target_sample_rate: int) -> tuple[np.ndarray, int]:
    """Load audio from WAV (always) and other formats if `soundfile` is available.

    Notes:
    - MP3 is typically NOT supported via soundfile/libsndfile.
    - For best compatibility, use WAV/FLAC inputs.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".wav" or sf is None:
        if suffix != ".wav" and sf is None:
            raise ValueError(
                f"Unsupported audio format: {suffix}. Install 'soundfile' or use WAV input. Path: {path}"
            )
        return load_wav(path, target_sample_rate)

    # Try soundfile for FLAC/OGG/etc.
    audio, sample_rate = sf.read(str(path), always_2d=False)
    audio = _to_mono_float32(np.asarray(audio))
    if int(sample_rate) != int(target_sample_rate):
        gcd = int(np.gcd(int(sample_rate), int(target_sample_rate)))
        audio = signal.resample_poly(audio, int(target_sample_rate) // gcd, int(sample_rate) // gcd).astype(np.float32)
        sample_rate = int(target_sample_rate)
    return audio, int(sample_rate)


def trim_silence(audio: np.ndarray, *, threshold: float = 1e-3) -> np.ndarray:
    """Trim leading/trailing silence using an amplitude threshold."""
    audio = audio.astype(np.float32, copy=False)
    idx = np.flatnonzero(np.abs(audio) > float(threshold))
    if len(idx) == 0:
        return audio
    return audio[int(idx[0]) : int(idx[-1]) + 1]


def remove_silence_top_db(audio: np.ndarray, *, top_db: float = 20.0) -> np.ndarray:
    """Librosa-like silence removal using a dB threshold relative to peak amplitude.

    This is a simple energy-based heuristic intended for notebook parity.
    """
    audio = audio.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(audio)) + 1e-9)
    # Convert dB below peak to linear amplitude threshold.
    threshold = peak * (10.0 ** (-float(top_db) / 20.0))
    idx = np.flatnonzero(np.abs(audio) > threshold)
    if len(idx) == 0:
        return audio

    # Keep contiguous segments above threshold.
    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[splits + 1]]
    ends = np.r_[idx[splits], idx[-1]]
    return np.concatenate([audio[int(s) : int(e) + 1] for s, e in zip(starts, ends)]).astype(np.float32)


def spectral_noise_subtract(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Simple spectral subtraction denoiser (fast, dependency-light)."""
    audio = audio.astype(np.float32, copy=False)
    if len(audio) < 8:
        return audio

    nperseg = max(32, min(512, len(audio)))
    noverlap = min(max(0, nperseg // 2), nperseg - 1)
    _, _, zxx = signal.stft(audio, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)

    # Estimate noise profile from first ~0.2s (or fewer bins if very short)
    noise_frames = max(1, int(0.2 * sample_rate / max(1, (nperseg - noverlap))))
    noise_profile = np.mean(np.abs(zxx[:, :noise_frames]), axis=1, keepdims=True)

    magnitude = np.abs(zxx)
    phase = np.angle(zxx)
    cleaned_mag = np.maximum(magnitude - 0.9 * noise_profile, 0.01 * magnitude)
    cleaned = signal.istft(cleaned_mag * np.exp(1j * phase), fs=sample_rate, nperseg=nperseg, noverlap=noverlap)[1]

    if len(cleaned) >= len(audio):
        return cleaned[: len(audio)].astype(np.float32)
    out = np.zeros(len(audio), dtype=np.float32)
    out[: len(cleaned)] = cleaned.astype(np.float32)
    return out


def rms_normalize(audio: np.ndarray, *, target_dbfs: float = -20.0) -> np.ndarray:
    """RMS normalisation to a target dBFS (approx)."""
    audio = audio.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(audio**2) + 1e-9))
    target_rms = 10 ** (float(target_dbfs) / 20.0)
    return (audio * (target_rms / (rms + 1e-9))).astype(np.float32)


def pad_or_crop(audio: np.ndarray, *, length: int) -> np.ndarray:
    """Center-crop or zero-pad to fixed length."""
    audio = audio.astype(np.float32, copy=False)
    if len(audio) >= length:
        start = (len(audio) - length) // 2
        return audio[start : start + length].astype(np.float32)
    out = np.zeros(length, dtype=np.float32)
    out[: len(audio)] = audio
    return out


@dataclass
class AudioPreprocessor:
    target_sample_rate: int = 16_000
    duration_seconds: float = 4.0
    trim_threshold: float = 1e-3
    target_dbfs: float = -20.0

    def load_audio(self, file_path: str | Path) -> tuple[np.ndarray, int]:
        """Backward-compatible loader used by notebooks."""
        return load_audio_any(file_path, self.target_sample_rate)

    def reduce_noise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Backward-compatible denoiser used by notebooks."""
        return spectral_noise_subtract(audio, sample_rate)

    def remove_silence(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """Backward-compatible silence removal used by notebooks."""
        return remove_silence_top_db(audio, top_db=float(top_db))

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Backward-compatible RMS normalizer used by notebooks."""
        return rms_normalize(audio, target_dbfs=self.target_dbfs)

    def preprocess(self, file_path: str | Path) -> tuple[np.ndarray, int]:
        """Full preprocessing pipeline (backward-compatible name)."""
        audio, sr = self.load_audio(file_path)
        audio = self.reduce_noise(audio, sr)
        audio = self.normalize_audio(audio)
        audio = self.remove_silence(audio)
        audio = pad_or_crop(audio, length=int(sr * float(self.duration_seconds)))
        return audio, sr

    def preprocess_wav(self, file_path: str | Path) -> tuple[np.ndarray, int]:
        """Alias kept for the CLI."""
        return self.preprocess(file_path)


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """Save float32 audio [-1,1] as int16 WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
    wavfile.write(path, int(sample_rate), (audio * 32_767.0).astype(np.int16))
