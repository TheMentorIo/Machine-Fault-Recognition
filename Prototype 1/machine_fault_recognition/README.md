# Machine Fault Recognition Package Guide

This README explains:
- The runtime flow
- What to run first
- What each folder is responsible for

## 1. Recommended Run Order

Use this order for large datasets:

1. Scan dataset structure
2. Extract and cache features
3. Train model branches and ensemble
4. Run inference on target test folder

Commands from project root:

```bash
python3 main.py scan --config configs/default.toml
python3 main.py features --config configs/default.toml
python3 main.py train --config configs/default.toml
python3 main.py infer --config configs/default.toml --state-path artifacts/pipeline_state.pkl --test-dir Student/Machine\ 1/Normal
```

Runtime control (pause/stop/resume):

```bash
python3 main.py control --set pause --config configs/default.toml
python3 main.py control --set stop --config configs/default.toml
python3 main.py control --set run --config configs/default.toml
```

You can also run only train. Training will try to reuse cached features from artifacts/feature_cache.npz and only re-extract if cache is missing or invalid.

## 2. Command Flow

CLI entrypoint:
- [cli/app.py](cli/app.py)

Primary stage handlers:
- [cli/train.py](cli/train.py)
- [cli/infer.py](cli/infer.py)

Execution details:
1. scan
- Builds dataset manifest via [training/pipeline.py](training/pipeline.py) -> scan_dataset
- Uses scanner logic in [data/scanner.py](data/scanner.py)
- Writes artifacts/dataset_manifest.csv

2. features
- Scans dataset
- Preprocesses audio and extracts two feature families in [training/pipeline.py](training/pipeline.py)
- Writes artifacts/feature_cache.npz

3. train
- Scans dataset
- Reuses feature cache if valid; otherwise recomputes
- Supports checkpointed resume for branch-level training steps
- Splits train/val/test with [data/splitter.py](data/splitter.py)
- Trains three branches from [models/](models)
- Calibrates, optimizes ensemble weights, evaluates metrics
- Writes:
  - artifacts/pipeline_state.pkl
  - artifacts/summary.json
  - artifacts/predictions.csv

4. infer
- Loads trained state from artifacts/pipeline_state.pkl
- Predicts per WAV file in resumable batches and saves:
  - artifacts/output/results.txt
  - artifacts/output/time.txt

## 3. Stage Status Tracking

Status tracking is implemented in [cli/app.py](cli/app.py).
Each command updates artifacts/stage_status.json with:
- started
- running
- paused
- stopped
- completed
- failed

If failed, an error message is stored under that stage key.

## 4. Resumable and Fault-Tolerant Execution

Persistent progress is stored per stage under artifacts/progress/:
- scan.json
- features.json
- train.json
- infer.json

Each stage stores enough checkpoint information to continue from the last successful batch or step:
- scan: resumes from next unprocessed batch of discovered files
- features: resumes from next feature batch; does not recompute completed batches
- train: resumes from saved branch checkpoints and calibration checkpoints
- infer: resumes from next prediction batch and partial outputs

Retry behavior:
- Batch-based stages include retry attempts for failed batches
- Progress and error details are saved before retry/fail decisions

Control behavior:
- pause: stage exits gracefully with progress preserved
- stop: stage exits gracefully with progress preserved
- run: normal execution; resumes from saved state

## 5. Configuration and Paths

Configuration model:
- [config.py](config.py)

Default config file (project root):
- [../configs/default.toml](../configs/default.toml)

Main path fields:
- input_dir (default Student)
- work_dir (default artifacts)
- model_path (default artifacts/pipeline_state.pkl)
- output_dir (default artifacts/output)

## 6. Folder-by-Folder Responsibilities

| Folder | What it contains | Main functionality |
|---|---|---|
| [calibration](calibration) | [temperature.py](calibration/temperature.py) | Temperature scaling for probability calibration |
| [cli](cli) | [app.py](cli/app.py), [train.py](cli/train.py), [infer.py](cli/infer.py), [__main__.py](cli/__main__.py) | Command parsing, stage orchestration, status tracking |
| [data](data) | [scanner.py](data/scanner.py), [splitter.py](data/splitter.py), [repositories.py](data/repositories.py), [contracts.py](data/contracts.py) | Dataset discovery, label extraction, split management, data contracts |
| [domain](domain) | [entities.py](domain/entities.py), [value_objects.py](domain/value_objects.py) | Domain-level data models |
| [ensemble](ensemble) | [optimizer.py](ensemble/optimizer.py), [voting.py](ensemble/voting.py) | Ensemble weight search, soft voting, confidence gating |
| [features](features) | [spectrogram.py](features/spectrogram.py), [handcrafted.py](features/handcrafted.py), [registry.py](features/registry.py) | Feature extraction (spectrogram + handcrafted) |
| [inference](inference) | [predictor.py](inference/predictor.py), [batcher.py](inference/batcher.py) | Inference-time feature path and prediction writing |
| [models](models) | [base.py](models/base.py), [branch_a.py](models/branch_a.py), [branch_b.py](models/branch_b.py), [branch_c.py](models/branch_c.py) | Three model branches and branch interfaces |
| [preprocessing](preprocessing) | [audio.py](preprocessing/audio.py), [augmentation.py](preprocessing/augmentation.py) | Audio preprocessing and augmentation utilities |
| [storage](storage) | [artifact_store.py](storage/artifact_store.py), [checkpoint.py](storage/checkpoint.py), [progress.py](storage/progress.py) | Artifact paths, checkpoint persistence, resumable progress and control signals |
| [training](training) | [pipeline.py](training/pipeline.py), [evaluators.py](training/evaluators.py), [callbacks.py](training/callbacks.py) | End-to-end training stages and evaluation |
| [utils](utils) | [logging.py](utils/logging.py), [seeding.py](utils/seeding.py), [device.py](utils/device.py) | Logging, reproducibility, device helpers |

Top-level package files:
- [__init__.py](__init__.py): package exports
- [config.py](config.py): global settings and path resolution

## 7. Typical Artifacts Produced

In artifacts/:
- dataset_manifest.csv
- feature_cache.npz
- pipeline_state.pkl
- summary.json
- predictions.csv
- stage_status.json
- control.json

In artifacts/progress/:
- scan.json
- features.json
- train.json
- infer.json

In artifacts/output/:
- results.txt
- time.txt

## 8. Practical Notes

- For large datasets, run scan and features once, then repeat train with cache reuse.
- If dataset labels or ordering change, training automatically discards invalid cache and rebuilds it.
- Progress bars are shown during scanning, feature extraction, and inference loops.
- To safely pause long runs, set control to pause, then set back to run to continue from the saved checkpoint.