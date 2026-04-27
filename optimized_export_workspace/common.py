from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT / "optimized_export_workspace"
RUNS_ROOT = WORKSPACE_ROOT / "runs"

DEFAULT_TTS_CHECKPOINT_DIR = REPO_ROOT / "MOSS-TTS-Nano-100M"
DEFAULT_TTS_ONNX_DIR = REPO_ROOT / "MOSS-TTS-Nano-100M-ONNX"
DEFAULT_CODEC_CHECKPOINT_DIR = REPO_ROOT / "MOSS-Audio-Tokenizer-Nano"
DEFAULT_CODEC_ONNX_DIR = REPO_ROOT / "MOSS-Audio-Tokenizer-Nano-ONNX"


def make_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def run_dir_for_id(run_id: str | None = None) -> Path:
    resolved = str(run_id or make_run_id()).strip()
    if not resolved:
        raise ValueError("run_id cannot be empty")
    if "/" in resolved or "\\" in resolved or resolved in {".", ".."}:
        raise ValueError(f"unsafe run_id: {run_id!r}")
    return RUNS_ROOT / resolved


def ensure_run_dirs(run_id: str | None = None) -> dict[str, Path]:
    run_dir = run_dir_for_id(run_id)
    paths = {
        "run": run_dir,
        "model": run_dir / "model",
        "tts": run_dir / "model" / "tts",
        "codec": run_dir / "model" / "codec",
        "audio": run_dir / "audio",
        "logs": run_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def repo_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)

