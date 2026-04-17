from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


class PauseRequested(Exception):
    pass


class StopRequested(Exception):
    pass


@dataclass
class ProgressStore:
    root: Path

    def __post_init__(self) -> None:
        self.progress_dir = self.root / "progress"
        self.progress_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, stage: str) -> Path:
        return self.progress_dir / f"{stage}.json"

    def load(self, stage: str) -> dict:
        path = self._path(stage)
        if not path.exists():
            return {
                "stage": stage,
                "status": "pending",
                "next_batch": 0,
                "total_batches": 0,
                "batch_size": 0,
                "retries": {},
                "last_error": "",
                "updated_at": self._now(),
            }
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, stage: str, state: dict) -> dict:
        state = dict(state)
        state["stage"] = stage
        state["updated_at"] = self._now()
        self._path(stage).write_text(json.dumps(state, indent=2), encoding="utf-8")
        return state

    def reset(self, stage: str) -> dict:
        state = self.load(stage)
        state.update(
            {
                "status": "pending",
                "next_batch": 0,
                "total_batches": 0,
                "batch_size": 0,
                "retries": {},
                "last_error": "",
            }
        )
        return self.save(stage, state)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


@dataclass
class ControlStore:
    root: Path

    def __post_init__(self) -> None:
        self.control_path = self.root / "control.json"

    def get_command(self) -> str:
        if not self.control_path.exists():
            return "run"
        try:
            payload = json.loads(self.control_path.read_text(encoding="utf-8"))
        except Exception:
            return "run"
        command = str(payload.get("command", "run")).strip().lower()
        return command if command in {"run", "pause", "stop"} else "run"

    def set_command(self, command: str) -> None:
        normalized = command.strip().lower()
        if normalized not in {"run", "pause", "stop"}:
            raise ValueError("command must be one of: run, pause, stop")
        self.control_path.parent.mkdir(parents=True, exist_ok=True)
        self.control_path.write_text(json.dumps({"command": normalized}, indent=2), encoding="utf-8")

    def check(self, stage: str, progress: ProgressStore, state: dict) -> None:
        command = self.get_command()
        if command == "pause":
            state["status"] = "paused"
            progress.save(stage, state)
            raise PauseRequested(f"Stage '{stage}' paused by control signal")
        if command == "stop":
            state["status"] = "stopped"
            progress.save(stage, state)
            raise StopRequested(f"Stage '{stage}' stopped by control signal")
