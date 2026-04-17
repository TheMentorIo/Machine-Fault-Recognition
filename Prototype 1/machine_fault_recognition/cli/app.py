from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ..config import load_config, resolve_work_dir
from ..storage.progress import ControlStore, PauseRequested, StopRequested
from .infer import infer
from .train import features, scan, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mfr", description="Machine Fault Recognition CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Scan the dataset and save a manifest")
    scan_parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))

    features_parser = subparsers.add_parser("features", help="Extract and cache features")
    features_parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))

    train_parser = subparsers.add_parser("train", help="Train the pipeline")
    train_parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))

    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    infer_parser.add_argument("--state-path", type=Path, default=None)
    infer_parser.add_argument("--test-dir", type=Path, default=None)

    control_parser = subparsers.add_parser("control", help="Set runtime control signal for resumable stages")
    control_parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    control_parser.add_argument("--set", dest="control_value", choices=["run", "pause", "stop"], required=True)

    return parser


def _stage_status_path(config_path: str | Path | None) -> Path:
    config = load_config(config_path)
    return resolve_work_dir(config) / "stage_status.json"


def _write_stage_status(stage: str, status: str, config_path: str | Path | None, error: str = "") -> None:
    status_path = _stage_status_path(config_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, dict[str, str]] = {}
    if status_path.exists():
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

    payload[stage] = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if error:
        payload[stage]["error"] = error

    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_with_status(stage: str, config_path: str | Path | None, runner) -> None:
    _write_stage_status(stage, "started", config_path)
    _write_stage_status(stage, "running", config_path)
    try:
        runner()
    except PauseRequested as exc:
        _write_stage_status(stage, "paused", config_path, error=str(exc))
        print(str(exc))
        return
    except StopRequested as exc:
        _write_stage_status(stage, "stopped", config_path, error=str(exc))
        print(str(exc))
        return
    except Exception as exc:
        _write_stage_status(stage, "failed", config_path, error=str(exc))
        raise
    _write_stage_status(stage, "completed", config_path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "scan":
        _run_with_status("scan", args.config, lambda: scan(args.config))
    elif args.command == "features":
        _run_with_status("features", args.config, lambda: features(args.config))
    elif args.command == "train":
        _run_with_status("train", args.config, lambda: train(args.config))
    elif args.command == "infer":
        _run_with_status("infer", args.config, lambda: infer(args.config, args.state_path, args.test_dir))
    elif args.command == "control":
        config = load_config(args.config)
        control = ControlStore(resolve_work_dir(config))
        control.set_command(args.control_value)
        print(f"Control signal set to '{args.control_value}' in {control.control_path}")


if __name__ == "__main__":
    main()
