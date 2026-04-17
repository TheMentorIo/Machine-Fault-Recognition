from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..features.handcrafted import HandcraftedFeatureExtractor
from ..features.spectrogram import SpectrogramExtractor
from ..preprocessing.audio import AudioPreprocessor


@dataclass
class AudioRepository:
    config: AppConfig

    def __post_init__(self) -> None:
        self.preprocessor = AudioPreprocessor(self.config.audio)
        self.spectrograms = SpectrogramExtractor(self.config.audio)
        self.handcrafted = HandcraftedFeatureExtractor(self.config.audio)

    def preprocess(self, path: Path, *, training: bool = False):
        return self.preprocessor.preprocess(path, training=training)

    def extract_spectrogram_features(self, audio: np.ndarray) -> np.ndarray:
        return self.spectrograms.features(audio)

    def extract_handcrafted_features(self, audio: np.ndarray) -> np.ndarray:
        return self.handcrafted.extract(audio)


@dataclass
class DataFrameRepository:
    frame: pd.DataFrame
