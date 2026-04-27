from __future__ import annotations

import argparse
import shutil
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

from optimized_export_workspace.common import (
    DEFAULT_CODEC_ONNX_DIR,
    DEFAULT_TTS_CHECKPOINT_DIR,
    DEFAULT_TTS_ONNX_DIR,
    ensure_run_dirs,
    read_json,
    repo_relative,
    write_json,
)
from optimized_export_workspace.export.graph_opt import (
    external_data_locations,
    load_output_shapes,
    save_with_last_hidden_output,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export optimized MOSS-TTS-Nano ONNX assets.")
    parser.add_argument("--run-id", default=None, help="Run id under optimized_export_workspace/runs.")
    parser.add_argument("--tts-onnx-dir", default=str(DEFAULT_TTS_ONNX_DIR), help="Official TTS ONNX directory.")
    parser.add_argument("--codec-onnx-dir", default=str(DEFAULT_CODEC_ONNX_DIR), help="Official codec ONNX directory.")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_TTS_CHECKPOINT_DIR), help="Local PyTorch checkpoint path.")
    return parser.parse_args(argv)


def _copy_required(source_dir: Path, target_dir: Path, names: Sequence[str]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target_dir / name)


def export_optimized_tts(
    *,
    run_id: str | None,
    tts_onnx_dir: str | Path,
    codec_onnx_dir: str | Path,
    checkpoint_dir: str | Path,
) -> dict[str, object]:
    paths = ensure_run_dirs(run_id)
    tts_source = Path(tts_onnx_dir).expanduser().resolve()
    codec_source = Path(codec_onnx_dir).expanduser().resolve()
    checkpoint_source = Path(checkpoint_dir).expanduser().resolve()
    tts_target = paths["tts"]
    codec_target = paths["codec"]

    _copy_required(
        tts_source,
        tts_target,
        [
            "tokenizer.model",
            "browser_poc_manifest.json",
            "tts_browser_onnx_meta.json",
            "moss_tts_global_shared.data",
            "moss_tts_local_shared.data",
            "moss_tts_local_decoder.onnx",
            "moss_tts_local_cached_step.onnx",
            "moss_tts_local_fixed_sampled_frame.onnx",
        ],
    )
    _copy_required(
        codec_source,
        codec_target,
        [
            "codec_browser_onnx_meta.json",
            "moss_audio_tokenizer_encode.onnx",
            "moss_audio_tokenizer_encode.data",
            "moss_audio_tokenizer_decode_full.onnx",
            "moss_audio_tokenizer_decode_step.onnx",
            "moss_audio_tokenizer_decode_shared.data",
        ],
    )

    prefill_path = save_with_last_hidden_output(
        tts_source / "moss_tts_prefill.onnx",
        tts_target / "moss_tts_prefill_last.onnx",
    )
    decode_path = save_with_last_hidden_output(
        tts_source / "moss_tts_decode_step.onnx",
        tts_target / "moss_tts_decode_step_last.onnx",
    )

    official_manifest = read_json(tts_source / "browser_poc_manifest.json")
    official_tts_meta = read_json(tts_source / "tts_browser_onnx_meta.json")
    optimized_meta = dict(official_tts_meta)
    optimized_meta["files"] = {
        **official_tts_meta["files"],
        "prefill": "moss_tts_prefill_last.onnx",
        "decode_step": "moss_tts_decode_step_last.onnx",
    }
    optimized_meta["onnx"] = {
        **official_tts_meta["onnx"],
        "prefill_output_names": [
            "global_hidden_last",
            *official_tts_meta["onnx"]["prefill_output_names"][1:],
        ],
        "decode_output_names": [
            "global_hidden_last",
            *official_tts_meta["onnx"]["decode_output_names"][1:],
        ],
    }
    write_json(tts_target / "optimized_tts_meta.json", optimized_meta)

    optimized_manifest = {
        "format_version": 1,
        "run_id": paths["run"].name,
        "source": {
            "checkpoint_dir": repo_relative(checkpoint_source),
            "official_tts_onnx_dir": repo_relative(tts_source),
            "official_codec_onnx_dir": repo_relative(codec_source),
        },
        "model_files": {
            "tts_meta": "optimized_tts_meta.json",
            "codec_meta": "../codec/codec_browser_onnx_meta.json",
            "tokenizer_model": "tokenizer.model",
        },
        "files": {
            "prefill": "moss_tts_prefill_last.onnx",
            "decode_step": "moss_tts_decode_step_last.onnx",
            "local_decoder": "moss_tts_local_decoder.onnx",
            "local_cached_step": "moss_tts_local_cached_step.onnx",
            "local_fixed_sampled_frame": "moss_tts_local_fixed_sampled_frame.onnx",
            "global_external_data": "moss_tts_global_shared.data",
            "local_external_data": "moss_tts_local_shared.data",
        },
        "tts_config": official_manifest["tts_config"],
        "prompt_templates": official_manifest["prompt_templates"],
        "generation_defaults": official_manifest["generation_defaults"],
        "builtin_voices": official_manifest["builtin_voices"],
        "text_samples": official_manifest["text_samples"],
    }
    manifest_path = write_json(tts_target / "optimized_manifest.json", optimized_manifest)

    report = {
        "run_id": paths["run"].name,
        "manifest": repo_relative(manifest_path),
        "prefill_output_shapes": load_output_shapes(prefill_path),
        "decode_output_shapes": load_output_shapes(decode_path),
        "prefill_external_data_locations": sorted(external_data_locations(prefill_path)),
        "decode_external_data_locations": sorted(external_data_locations(decode_path)),
    }
    write_json(paths["run"] / "export_report.json", report)
    write_json(
        paths["logs"] / "export_log.json",
        {
            "event": "optimized_tts_export",
            "status": "success",
            **report,
        },
    )
    return report


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    report = export_optimized_tts(
        run_id=args.run_id,
        tts_onnx_dir=args.tts_onnx_dir,
        codec_onnx_dir=args.codec_onnx_dir,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(f"optimized export complete: {report['manifest']}")
    return report


if __name__ == "__main__":
    main()
