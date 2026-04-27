from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


def _ensure_repo_root_on_path() -> Path:
    current = Path(__file__).resolve()
    repo_root = current.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


_ensure_repo_root_on_path()

from optimized_export_workspace.common import run_dir_for_id


UNUSED_TTS_FILES = {
    "browser_poc_manifest.json",
    "tts_browser_onnx_meta.json",
    "moss_tts_local_decoder.onnx",
    "moss_tts_local_cached_step.onnx",
    "moss_tts_prefill_last_int8.onnx",
    "moss_tts_decode_step_last_int8.onnx",
    "moss_tts_local_fixed_sampled_frame_int8.onnx",
}
UNUSED_CODEC_FILES = {
    "moss_audio_tokenizer_decode_step.onnx",
}
REQUIRED_TTS_FILES = {
    "optimized_manifest.json",
    "optimized_tts_meta.json",
    "tokenizer.model",
    "moss_tts_prefill_last.onnx",
    "moss_tts_decode_step_last.onnx",
    "moss_tts_local_fixed_sampled_frame.onnx",
    "moss_tts_global_shared.data",
    "moss_tts_local_shared.data",
}
REQUIRED_CODEC_FILES = {
    "codec_browser_onnx_meta.json",
    "moss_audio_tokenizer_encode.onnx",
    "moss_audio_tokenizer_encode.data",
    "moss_audio_tokenizer_decode_full.onnx",
    "moss_audio_tokenizer_decode_shared.data",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check optimized export package is slim.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("slim", "fp16"), default="slim")
    return parser.parse_args(argv)


def directory_file_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.iterdir() if item.is_file())


def assert_slim_export(run_id: str, mode: str = "slim") -> dict[str, object]:
    run_dir = run_dir_for_id(run_id)
    tts_dir = run_dir / "model" / "tts"
    codec_dir = run_dir / "model" / "codec"
    assert tts_dir.is_dir(), f"missing TTS dir: {tts_dir}"
    assert codec_dir.is_dir(), f"missing codec dir: {codec_dir}"

    tts_names = {item.name for item in tts_dir.iterdir() if item.is_file()}
    codec_names = {item.name for item in codec_dir.iterdir() if item.is_file()}
    if mode == "fp16":
        required_tts_files = {
            "optimized_manifest.json",
            "optimized_tts_meta.json",
            "tokenizer.model",
            "moss_tts_prefill_last_fp16.onnx",
            "moss_tts_decode_step_last_fp16.onnx",
            "moss_tts_local_fixed_sampled_frame_fp16.onnx",
            "moss_tts_global_fp16_shared.data",
            "moss_tts_local_fixed_sampled_frame_fp16.data",
        }
    else:
        required_tts_files = REQUIRED_TTS_FILES
    assert required_tts_files <= tts_names, f"missing TTS files: {sorted(required_tts_files - tts_names)}"
    assert REQUIRED_CODEC_FILES <= codec_names, f"missing codec files: {sorted(REQUIRED_CODEC_FILES - codec_names)}"
    assert not (UNUSED_TTS_FILES & tts_names), f"unused TTS files copied: {sorted(UNUSED_TTS_FILES & tts_names)}"
    assert not (UNUSED_CODEC_FILES & codec_names), f"unused codec files copied: {sorted(UNUSED_CODEC_FILES & codec_names)}"

    manifest = json.loads((tts_dir / "optimized_manifest.json").read_text(encoding="utf-8"))
    assert manifest["package"]["mode"] == mode
    assert manifest["model_files"]["codec_meta"] == "../codec/codec_browser_onnx_meta.json"
    report = {
        "run_id": run_id,
        "tts_file_count": len(tts_names),
        "codec_file_count": len(codec_names),
        "tts_size_bytes": directory_file_size(tts_dir),
        "codec_size_bytes": directory_file_size(codec_dir),
    }
    (run_dir / "slim_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    report = assert_slim_export(args.run_id, mode=args.mode)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
