# Machine Fault Recognition: Optimized End-to-End Plan

This document defines an accuracy-first, speed-aware architecture for large-scale machine-fault classification.

Prototype 1 is provided in two synchronized forms:

- `kaggle_prototype.ipynb` for Kaggle execution
- `architecture_pipeline.py` for a normal Python entry point

For a production-grade OOP structure, see [docs/Architecture/Production OOP Architecture.md](docs/Architecture/Production%20OOP%20Architecture.md).

Both follow the restricted architecture below and avoid extra baseline-only components.

## Run

The runnable code now lives in the [machine_fault_recognition](machine_fault_recognition) package.

For a folder-by-folder flow guide and run order, see [machine_fault_recognition/README.md](machine_fault_recognition/README.md).

From the project root, you can start training with [main.py](main.py) or the CLI module:

```bash
python main.py scan --config configs/default.toml
python main.py features --config configs/default.toml
python main.py train --config configs/default.toml
python -m machine_fault_recognition.cli.app train --config configs/default.toml
```

To generate predictions from a saved pipeline state:

```bash
python main.py infer --config configs/default.toml --state-path artifacts/pipeline_state.pkl --test-dir Student
```

Artifacts are written under [artifacts/](artifacts) by default.

## Targets

- Accuracy target: 95% to 97%+ on held-out test data
- Inference target: under 0.5 s per file on a modern GPU stack
- Throughput target: stable batched inference with deterministic outputs

## Stage 1: Preprocessing (Cython-compiled)

Implement the hot preprocessing path in `preprocess.pyx` and compile with `cythonize`.

### Scope of Cython hot path

- Resampling to a fixed sample rate
- Spectral noise subtraction
- Loudness normalization to -23 LUFS
- Silence trimming

### Shape standardization

- Pad or crop all clips to exactly 4.0 seconds
- Keep tensor shapes identical across all branches
- Enable batching instead of file-by-file execution

### Why this stage matters

- Expected speedup: about 3x to 5x versus pure Python preprocessing
- Reduced Python overhead in tight loops
- Better cache and batching behavior downstream

## Stage 2: Augmentation (training only)

Use online augmentation in the training dataloader only. Do not apply augmentation to validation or test sets.

### Augmentation policy

- `AddGaussianNoise`
- `TimeStretch` in the range ±10%
- `PitchShift` in the range ±2 semitones
- `SpecAugment` (time masking and frequency masking)
- `Mixup` between samples from the same machine domain

### Goal

Improve robustness to microphone quality differences, background noise, and volume shifts while preserving honest evaluation.

## Stage 3: Three parallel model branches

All branches consume the same preprocessed audio but use different representations.

### Branch A: Spectrogram vision model (heavyweight)

- Input channels: log-mel, CQT, delta-mel
- Backbone: EfficientNet-B2 pretrained on AudioSet
- Expected standalone accuracy: approximately 94% to 96%

Strength: highest single-model ceiling on rich time-frequency structure.

### Branch B: Conformer branch

- Small CNN front-end for local pattern extraction
- Transformer encoder for long-range temporal context

Strength: captures periodic rhythms and sequence-level fault dynamics that CNN-only models may miss.

### Branch C: Statistical features + classical ensemble (CPU anchor)

- Feature vector (~300 dims):
	- MFCC mean/std/skew
	- Spectral centroid, rolloff, flux
	- ZCR, RMS, chroma
- Models: XGBoost + LightGBM + calibrated SVM

Strength: fast, interpretable, stable under uncertainty, and complementary to deep branches.

## Stage 4: Calibration and weighted soft-vote ensemble

Calibrate each branch independently before ensembling.

### Calibration

- Use temperature scaling on validation predictions
- Apply calibrated probabilities at inference time

Reason: uncalibrated softmax outputs are often overconfident and can degrade soft-vote quality.

### Weighted fusion

- Optimize branch weights `(wA, wB, wC)` on validation accuracy
- Use Nelder-Mead from initial weights `[0.40, 0.40, 0.20]`
- Constraint: weights are non-negative and sum to 1

## Stage 5: Confidence-gated agent decision

After weighted ensemble, apply a confidence gate:

- Compute `max(prob_vector)`
- If confidence is `>= 0.60`, output ensemble top class
- If confidence is `< 0.60`, route to the single branch with highest individual confidence on that sample

This gate recovers hard cases where averaging uncertain outputs hurts final decisions.

### Tuning

- Validate threshold in a sweep (for example 0.50 to 0.75)
- Pick threshold by validation F1 or accuracy based on competition metric

## Stage 6: Speed optimization stack

### Runtime optimizations

- Compile `preprocess.pyx` with typed memoryviews for NumPy arrays
- Export Branch A and Branch B to ONNX
- Apply TensorRT INT8 quantization (target: <0.5% accuracy loss)
- Run Branches A, B, C concurrently with `ThreadPoolExecutor`
- Precompute and memory-map spectrogram tensors to reduce per-file I/O
- Warm up all models with dummy forward passes at startup
- Micro-batch 8 files for GPU branches, then restore original output order

### Timing protocol

- Start inference timer after file read
- Report only pipeline compute time in `time.txt`

## Validation Protocol (recommended)

- Use machine-aware or environment-aware split to avoid leakage
- Track per-class precision, recall, and F1 in addition to accuracy
- Produce reliability diagrams and expected calibration error after scaling
- Run ablations:
	- Branch A only
	- A+B
	- A+B+C
	- A+B+C with confidence gate

## Expected Outcome

- Ensemble improvement over Branch A alone: typically +1% to +2%
- Confidence gate gain on hard samples: often about +0.5%
- Overall target: 95% to 97%+ without sacrificing runtime competitiveness