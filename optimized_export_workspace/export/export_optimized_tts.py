from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import onnx


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
    export_ort_optimized_model,
    external_data_locations,
    graph_stats,
    load_output_shapes,
    save_with_last_hidden_output,
    simplify_model_to_external_data,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export optimized MOSS-TTS-Nano ONNX assets.")
    parser.add_argument("--run-id", default=None, help="Run id under optimized_export_workspace/runs.")
    parser.add_argument("--tts-onnx-dir", default=str(DEFAULT_TTS_ONNX_DIR), help="Official TTS ONNX directory.")
    parser.add_argument("--codec-onnx-dir", default=str(DEFAULT_CODEC_ONNX_DIR), help="Official codec ONNX directory.")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_TTS_CHECKPOINT_DIR), help="Local PyTorch checkpoint path.")
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16"),
        default="fp32",
        help="TTS graph weight precision. fp16 reduces size and keeps graph I/O as fp32.",
    )
    parser.add_argument(
        "--codec-precision",
        choices=("fp32", "fp16"),
        default=None,
        help="Codec graph weight precision. Defaults to the same value as --precision.",
    )
    parser.add_argument(
        "--offline-ort-optimize",
        action="store_true",
        help="Save ORT_ENABLE_ALL optimized TTS graphs back into the exported package.",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Run onnx-simplifier on the local fixed-frame graph before optional ORT optimization.",
    )
    return parser.parse_args(argv)


def _copy_required(source_dir: Path, target_dir: Path, names: Sequence[str]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target_dir / name)


def _reset_generated_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _directory_file_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.iterdir() if item.is_file())


def _build_slim_tts_meta(
    official_tts_meta: dict[str, Any],
    *,
    prefill_file: str,
    decode_file: str,
    local_fixed_file: str,
) -> dict[str, Any]:
    onnx_meta = official_tts_meta["onnx"]
    return {
        "format_version": official_tts_meta.get("format_version", 1),
        "checkpoint_path": official_tts_meta.get("checkpoint_path", ""),
        "files": {
            "prefill": prefill_file,
            "decode_step": decode_file,
            "local_fixed_sampled_frame": local_fixed_file,
        },
        "model_config": official_tts_meta["model_config"],
        "onnx": {
            "opset": onnx_meta["opset"],
            "prefill_output_names": [
                "global_hidden_last",
                *onnx_meta["prefill_output_names"][1:],
            ],
            "decode_input_names": onnx_meta["decode_input_names"],
            "decode_output_names": [
                "global_hidden_last",
                *onnx_meta["decode_output_names"][1:],
            ],
            "local_fixed_sampled_frame_input_names": onnx_meta["local_fixed_sampled_frame_input_names"],
            "local_fixed_sampled_frame_output_names": onnx_meta["local_fixed_sampled_frame_output_names"],
            "fixed_sampled_frame_constants": onnx_meta["fixed_sampled_frame_constants"],
        },
    }


def _save_fp16_model(
    source_path: Path,
    target_path: Path,
    external_data_name: str,
    *,
    disable_shape_infer: bool = True,
    block_rope: bool = True,
) -> Path:
    try:
        from onnxconverter_common import float16
    except ImportError as exc:
        raise RuntimeError("onnxconverter-common is required for --precision fp16") from exc
    model = onnx.load(str(source_path), load_external_data=True)
    blocked_ops = list(float16.DEFAULT_OP_BLOCK_LIST)
    producer_by_output = {
        output_name: node.name
        for node in model.graph.node
        for output_name in node.output
        if node.name
    }
    blocked_node_names = {
        node.name
        for node in model.graph.node
        if node.name and (node.op_type in blocked_ops or (block_rope and "/rope/" in node.name))
    }
    for node in model.graph.node:
        if node.op_type in blocked_ops or (block_rope and "/rope/" in node.name):
            blocked_node_names.update(
                producer_by_output[input_name]
                for input_name in node.input
                if input_name in producer_by_output
            )
    converted = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=disable_shape_infer,
        op_block_list=blocked_ops,
        node_block_list=sorted(blocked_node_names) if blocked_node_names else None,
    )
    external_data_path = target_path.parent / external_data_name
    if external_data_path.exists():
        external_data_path.unlink()
    onnx.save_model(
        converted,
        str(target_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_name,
        size_threshold=1024,
    )
    return target_path


def _rewrite_external_data_location(model_path: Path, old_location: str, new_location: str) -> None:
    model = onnx.load(str(model_path), load_external_data=False)
    for initializer in model.graph.initializer:
        for entry in initializer.external_data:
            if entry.key == "location" and entry.value == old_location:
                entry.value = new_location
    onnx.save_model(model, str(model_path), save_as_external_data=False)


def export_optimized_tts(
    *,
    run_id: str | None,
    tts_onnx_dir: str | Path,
    codec_onnx_dir: str | Path,
    checkpoint_dir: str | Path,
    precision: str = "fp32",
    codec_precision: str | None = None,
    offline_ort_optimize: bool = False,
    simplify: bool = False,
) -> dict[str, object]:
    if precision not in {"fp32", "fp16"}:
        raise ValueError(f"unsupported precision: {precision}")
    codec_precision = codec_precision or precision
    if codec_precision not in {"fp32", "fp16"}:
        raise ValueError(f"unsupported codec precision: {codec_precision}")
    paths = ensure_run_dirs(run_id)
    tts_source = Path(tts_onnx_dir).expanduser().resolve()
    codec_source = Path(codec_onnx_dir).expanduser().resolve()
    checkpoint_source = Path(checkpoint_dir).expanduser().resolve()
    tts_target = paths["tts"]
    codec_target = paths["codec"]
    _reset_generated_dir(tts_target)
    _reset_generated_dir(codec_target)

    _copy_required(
        tts_source,
        tts_target,
        [
            "tokenizer.model",
            "moss_tts_global_shared.data",
            "moss_tts_local_shared.data",
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
            "moss_audio_tokenizer_decode_shared.data",
        ],
    )
    codec_encode_path = codec_target / "moss_audio_tokenizer_encode.onnx"
    codec_decode_path = codec_target / "moss_audio_tokenizer_decode_full.onnx"
    if codec_precision == "fp16":
        codec_encode_tmp = codec_target / "moss_audio_tokenizer_encode_fp16.tmp.onnx"
        codec_decode_tmp = codec_target / "moss_audio_tokenizer_decode_full_fp16.tmp.onnx"
        _save_fp16_model(
            codec_encode_path,
            codec_encode_tmp,
            "moss_audio_tokenizer_encode.data",
            disable_shape_infer=False,
            block_rope=False,
        )
        _save_fp16_model(
            codec_decode_path,
            codec_decode_tmp,
            "moss_audio_tokenizer_decode_shared.data",
            disable_shape_infer=False,
            block_rope=False,
        )
        codec_encode_tmp.replace(codec_encode_path)
        codec_decode_tmp.replace(codec_decode_path)

    fp32_prefill_path = save_with_last_hidden_output(
        tts_source / "moss_tts_prefill.onnx",
        tts_target / "moss_tts_prefill_last.onnx",
    )
    fp32_decode_path = save_with_last_hidden_output(
        tts_source / "moss_tts_decode_step.onnx",
        tts_target / "moss_tts_decode_step_last.onnx",
    )
    prefill_file = "moss_tts_prefill_last.onnx"
    decode_file = "moss_tts_decode_step_last.onnx"
    local_fixed_file = "moss_tts_local_fixed_sampled_frame.onnx"
    local_external_data_file = "moss_tts_local_shared.data"
    if precision == "fp16":
        prefill_file = "moss_tts_prefill_last_fp16.onnx"
        decode_file = "moss_tts_decode_step_last_fp16.onnx"
        local_fixed_file = "moss_tts_local_fixed_sampled_frame_fp16.onnx"
        global_fp16_data = "moss_tts_global_fp16_shared.data"
        local_fp16_data = "moss_tts_local_fixed_sampled_frame_fp16.data"
        local_external_data_file = local_fp16_data
        prefill_path = _save_fp16_model(fp32_prefill_path, tts_target / prefill_file, global_fp16_data)
        decode_temp_data = "moss_tts_decode_step_last_fp16.tmp.data"
        decode_path = _save_fp16_model(fp32_decode_path, tts_target / decode_file, decode_temp_data)
        if filecmp.cmp(tts_target / global_fp16_data, tts_target / decode_temp_data, shallow=False):
            (tts_target / decode_temp_data).unlink()
            _rewrite_external_data_location(decode_path, decode_temp_data, global_fp16_data)
        else:
            global_fp16_data = decode_temp_data
        local_fixed_path = _save_fp16_model(
            tts_target / "moss_tts_local_fixed_sampled_frame.onnx",
            tts_target / local_fixed_file,
            local_fp16_data,
        )
        for removable in [
            fp32_prefill_path,
            fp32_decode_path,
            tts_target / "moss_tts_local_fixed_sampled_frame.onnx",
            tts_target / "moss_tts_global_shared.data",
            tts_target / "moss_tts_local_shared.data",
        ]:
            if removable.exists():
                removable.unlink()
    else:
        prefill_path = fp32_prefill_path
        decode_path = fp32_decode_path
        local_fixed_path = tts_target / local_fixed_file

    graph_reports: dict[str, Any] = {}
    graph_paths = {
        "prefill": prefill_path,
        "decode_step": decode_path,
        "local_fixed_sampled_frame": local_fixed_path,
        "codec_encode": codec_encode_path,
        "codec_decode_full": codec_decode_path,
    }
    initial_graph_stats = {
        graph_name: graph_stats(graph_path)
        for graph_name, graph_path in graph_paths.items()
    }
    if simplify:
        local_simplified_file = (
            "moss_tts_local_fixed_sampled_frame_simplified.onnx"
            if precision == "fp32"
            else "moss_tts_local_fixed_sampled_frame_fp16_simplified.onnx"
        )
        local_simplified_data = (
            "moss_tts_local_fixed_sampled_frame_simplified.data"
            if precision == "fp32"
            else "moss_tts_local_fixed_sampled_frame_fp16_simplified.data"
        )
        simplify_report = simplify_model_to_external_data(
            local_fixed_path,
            tts_target / local_simplified_file,
            external_data_name=local_simplified_data,
        )
        if simplify_report["status"] == "success":
            if local_fixed_path.exists():
                local_fixed_path.unlink()
            old_local_data = tts_target / (
                "moss_tts_local_fixed_sampled_frame_fp16.data"
                if precision == "fp16"
                else "moss_tts_local_shared.data"
            )
            if old_local_data.exists():
                old_local_data.unlink()
            local_fixed_file = local_simplified_file
            local_external_data_file = local_simplified_data
            local_fixed_path = tts_target / local_fixed_file
            graph_paths["local_fixed_sampled_frame"] = local_fixed_path
        graph_reports["local_fixed_sampled_frame"] = {
            "simplify": simplify_report,
        }
    actual_simplified = (
        graph_reports
        .get("local_fixed_sampled_frame", {})
        .get("simplify", {})
        .get("status") == "success"
    )

    if offline_ort_optimize:
        for graph_name, graph_path in list(graph_paths.items()):
            graph_reports.setdefault(graph_name, {})
            graph_reports[graph_name]["ort"] = export_ort_optimized_model(graph_path, thread_count=1)

    for graph_name, graph_path in graph_paths.items():
        graph_reports.setdefault(graph_name, {})
        graph_reports[graph_name].setdefault(
            "simplify",
            {
                "enabled": simplify,
                "status": "skipped",
                "reason": (
                    "only TTS local_fixed_sampled_frame is simplified"
                    if graph_name != "local_fixed_sampled_frame"
                    else "not requested"
                ),
            },
        )
        graph_reports[graph_name].setdefault(
            "ort",
            {
                "enabled": offline_ort_optimize,
                "status": "skipped",
                "reason": "not requested",
            },
        )
        # Capture stats after all requested transformations have been applied.
        final_stats = graph_stats(graph_path)
        source_stats = initial_graph_stats[graph_name]
        if graph_reports[graph_name]["simplify"].get("before_nodes"):
            source_nodes = graph_reports[graph_name]["simplify"]["before_nodes"]
        elif graph_reports[graph_name]["ort"].get("before_nodes"):
            source_nodes = graph_reports[graph_name]["ort"]["before_nodes"]
        else:
            source_nodes = final_stats["nodes"]
        graph_reports[graph_name].update(
            {
                "source_nodes": source_nodes,
                "final_nodes": final_stats["nodes"],
                "final_size_bytes": final_stats["size_bytes"],
                "source_op_counts": source_stats["op_counts"],
                "final_op_counts": final_stats["op_counts"],
            }
        )

    official_manifest = read_json(tts_source / "browser_poc_manifest.json")
    official_tts_meta = read_json(tts_source / "tts_browser_onnx_meta.json")
    optimized_meta = _build_slim_tts_meta(
        official_tts_meta,
        prefill_file=prefill_file,
        decode_file=decode_file,
        local_fixed_file=local_fixed_file,
    )
    write_json(tts_target / "optimized_tts_meta.json", optimized_meta)

    optimized_manifest = {
        "format_version": 1,
        "run_id": paths["run"].name,
        "package": {
            "mode": "fp16" if precision == "fp16" else "slim",
            "lossless": precision == "fp32",
            "precision": precision,
            "codec_precision": codec_precision,
            "offline_ort_optimized": offline_ort_optimize,
            "simplify_requested": simplify,
            "simplified": actual_simplified,
            "omitted_tts_files": [
                "browser_poc_manifest.json",
                "tts_browser_onnx_meta.json",
                "moss_tts_local_decoder.onnx",
                "moss_tts_local_cached_step.onnx",
            ],
            "omitted_codec_files": [
                "moss_audio_tokenizer_decode_step.onnx",
            ],
        },
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
            "prefill": prefill_file,
            "decode_step": decode_file,
            "local_fixed_sampled_frame": local_fixed_file,
            **(
                {
                    "global_external_data": "moss_tts_global_fp16_shared.data",
                    "local_external_data": local_external_data_file,
                }
                if precision == "fp16"
                else {
                    "global_external_data": "moss_tts_global_shared.data",
                    "local_external_data": local_external_data_file,
                }
            ),
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
        "local_fixed_sampled_frame_external_data_locations": sorted(external_data_locations(local_fixed_path)),
        "codec_encode_external_data_locations": sorted(external_data_locations(codec_encode_path)),
        "codec_decode_full_external_data_locations": sorted(external_data_locations(codec_decode_path)),
        "tts_size_bytes": _directory_file_size(tts_target),
        "codec_size_bytes": _directory_file_size(codec_target),
        "precision": precision,
        "codec_precision": codec_precision,
        "offline_ort_optimized": offline_ort_optimize,
        "simplify_requested": simplify,
        "simplified": actual_simplified,
    }
    write_json(paths["run"] / "export_report.json", report)
    write_json(
        paths["run"] / "graph_optimization_report.json",
        {
            "run_id": paths["run"].name,
            "precision": precision,
            "codec_precision": codec_precision,
            "offline_ort_optimized": offline_ort_optimize,
            "simplify_requested": simplify,
            "simplified": actual_simplified,
            "graphs": graph_reports,
        },
    )
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
        precision=args.precision,
        codec_precision=args.codec_precision,
        offline_ort_optimize=args.offline_ort_optimize,
        simplify=args.simplify,
    )
    print(f"optimized export complete: {report['manifest']}")
    return report


if __name__ == "__main__":
    main()
