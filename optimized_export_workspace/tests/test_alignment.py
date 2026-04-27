from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM


def _ensure_repo_root_on_path() -> Path:
    current = Path(__file__).resolve()
    repo_root = current.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = _ensure_repo_root_on_path()

from optimized_export_workspace.common import DEFAULT_TTS_CHECKPOINT_DIR, run_dir_for_id
from optimized_export_workspace.export.graph_opt import external_data_locations, load_output_shapes
from optimized_export_workspace.runtime.optimized_runtime import OptimizedTtsRuntime
from ort_cpu_runtime import OrtCpuRuntime


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate optimized MOSS-TTS-Nano ONNX export parity.")
    parser.add_argument("--run-id", required=True, help="Run id under optimized_export_workspace/runs.")
    parser.add_argument("--max-new-frames", type=int, default=8, help="Reserved for generation checks.")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for hidden-state comparisons.")
    parser.add_argument("--skip-pytorch", action="store_true", help="Skip official PyTorch prefill comparison.")
    return parser.parse_args(argv)


def assert_export_layout(run_id: str) -> dict[str, Path]:
    run_dir = run_dir_for_id(run_id)
    tts_dir = run_dir / "model" / "tts"
    manifest_path = tts_dir / "optimized_manifest.json"
    assert manifest_path.is_file(), f"missing optimized manifest: {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["files"]
    paths = {
        "run_dir": run_dir,
        "manifest": manifest_path,
        "prefill": tts_dir / files["prefill"],
        "decode": tts_dir / files["decode_step"],
    }
    if "global_external_data" in files:
        paths["global_data"] = tts_dir / files["global_external_data"]
    if "local_external_data" in files:
        paths["local_data"] = tts_dir / files["local_external_data"]
    for label, path in paths.items():
        if label in {"run_dir", "manifest"}:
            continue
        assert path.is_file(), f"missing {label}: {path}"
    return paths


def assert_last_hidden_outputs(paths: dict[str, Path]) -> None:
    prefill_shapes = load_output_shapes(paths["prefill"])
    decode_shapes = load_output_shapes(paths["decode"])
    assert prefill_shapes["global_hidden_last"][-1] == 768
    assert decode_shapes["global_hidden_last"][-1] == 768
    assert len(prefill_shapes["global_hidden_last"]) == 2
    assert len(decode_shapes["global_hidden_last"]) == 2


def assert_shared_external_data(paths: dict[str, Path]) -> None:
    if "global_data" not in paths:
        return
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    expected_global = manifest["files"]["global_external_data"]
    prefill_locations = external_data_locations(paths["prefill"])
    decode_locations = external_data_locations(paths["decode"])
    assert prefill_locations == {expected_global}
    assert decode_locations == {expected_global}


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def _flatten3d_int32(nested: list[list[list[int]]]) -> tuple[np.ndarray, list[int]]:
    dim0 = len(nested)
    dim1 = len(nested[0])
    dim2 = len(nested[0][0])
    data = np.asarray([value for plane in nested for row in plane for value in row], dtype=np.int32)
    return data, [dim0, dim1, dim2]


def _flatten2d_int32(nested: list[list[int]]) -> tuple[np.ndarray, list[int]]:
    dim0 = len(nested)
    dim1 = len(nested[0])
    data = np.asarray([value for row in nested for value in row], dtype=np.int32)
    return data, [dim0, dim1]


def _max_mean_abs(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    diff = np.abs(np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32))
    return float(np.max(diff)), float(np.mean(diff))


def _official_tts_dir(paths: dict[str, Path]) -> Path:
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    return (REPO_ROOT / manifest["source"]["official_tts_onnx_dir"]).resolve()


def compare_prefill_and_decode_last_hidden(paths: dict[str, Path], atol: float) -> dict[str, float]:
    official_tts_dir = _official_tts_dir(paths)
    optimized_runtime = OptimizedTtsRuntime(paths["run_dir"].name, seed=1234)
    voice = optimized_runtime.list_builtin_voices()[0]
    text_sample = optimized_runtime.manifest["text_samples"][0]
    request_rows = optimized_runtime.build_voice_clone_request_rows(
        voice["prompt_audio_codes"],
        text_sample["text_token_ids"],
    )
    prefill_ids, prefill_dims = _flatten3d_int32([request_rows["inputIds"]])
    prefill_mask, prefill_mask_dims = _flatten2d_int32(request_rows["attentionMask"])
    prefill_feeds = {
        "input_ids": prefill_ids.reshape(prefill_dims),
        "attention_mask": prefill_mask.reshape(prefill_mask_dims),
    }

    official_prefill_session = _session(official_tts_dir / "moss_tts_prefill.onnx")
    optimized_prefill_session = _session(paths["prefill"])
    official_prefill_outputs = official_prefill_session.run(None, prefill_feeds)
    optimized_prefill_outputs = optimized_prefill_session.run(None, prefill_feeds)
    official_prefill_named = dict(
        zip([output.name for output in official_prefill_session.get_outputs()], official_prefill_outputs, strict=True)
    )
    optimized_prefill_named = dict(
        zip([output.name for output in optimized_prefill_session.get_outputs()], optimized_prefill_outputs, strict=True)
    )
    prefill_max, prefill_mean = _max_mean_abs(
        official_prefill_named["global_hidden"][:, -1, :],
        optimized_prefill_named["global_hidden_last"],
    )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    lossless = bool(manifest.get("package", {}).get("lossless", True))
    if lossless:
        assert prefill_max <= atol, f"prefill max_abs_error {prefill_max} > {atol}"

    n_vq = int(optimized_runtime.manifest["tts_config"]["n_vq"])
    row_width = n_vq + 1
    decode_row = np.full(
        (1, 1, row_width),
        int(optimized_runtime.manifest["tts_config"]["audio_pad_token_id"]),
        dtype=np.int32,
    )
    decode_row[0, 0, 0] = int(optimized_runtime.manifest["tts_config"]["audio_assistant_slot_token_id"])
    for index in range(n_vq):
        decode_row[0, 0, index + 1] = index
    official_decode_session = _session(official_tts_dir / "moss_tts_decode_step.onnx")
    optimized_decode_session = _session(paths["decode"])
    past_valid_length = sum(int(item) for item in request_rows["attentionMask"][0])
    decode_feeds = {
        "input_ids": decode_row,
        "past_valid_lengths": np.asarray([past_valid_length], dtype=np.int32),
    }
    for input_name in optimized_runtime.tts_meta["onnx"]["decode_input_names"][2:]:
        present_name = input_name.replace("past_", "present_")
        decode_feeds[input_name] = official_prefill_named[present_name]
    official_decode_outputs = official_decode_session.run(None, decode_feeds)
    optimized_decode_outputs = optimized_decode_session.run(None, decode_feeds)
    official_decode_named = dict(
        zip([output.name for output in official_decode_session.get_outputs()], official_decode_outputs, strict=True)
    )
    optimized_decode_named = dict(
        zip([output.name for output in optimized_decode_session.get_outputs()], optimized_decode_outputs, strict=True)
    )
    decode_max, decode_mean = _max_mean_abs(
        official_decode_named["global_hidden"][:, -1, :],
        optimized_decode_named["global_hidden_last"],
    )
    if lossless:
        assert decode_max <= atol, f"decode max_abs_error {decode_max} > {atol}"
    return {
        "prefill_max_abs_error": prefill_max,
        "prefill_mean_abs_error": prefill_mean,
        "prefill_within_tolerance": prefill_max <= atol,
        "decode_max_abs_error": decode_max,
        "decode_mean_abs_error": decode_mean,
        "decode_within_tolerance": decode_max <= atol,
    }


def compare_short_generation(run_id: str, max_new_frames: int) -> dict[str, object]:
    official_runtime = OrtCpuRuntime(REPO_ROOT, max_new_frames=max_new_frames, sample_mode="fixed")
    official_runtime.rng = np.random.default_rng(1234)
    optimized_runtime = OptimizedTtsRuntime(run_id, seed=1234)
    voice = optimized_runtime.list_builtin_voices()[0]
    text_sample = optimized_runtime.manifest["text_samples"][0]
    official_request_rows = official_runtime.build_voice_clone_request_rows(
        voice["prompt_audio_codes"],
        text_sample["text_token_ids"],
    )
    optimized_request_rows = optimized_runtime.build_voice_clone_request_rows(
        voice["prompt_audio_codes"],
        text_sample["text_token_ids"],
    )
    official_frames = official_runtime.generate_audio_frames(official_request_rows)
    optimized_frames = optimized_runtime.generate_audio_frames(
        optimized_request_rows,
        max_new_frames=max_new_frames,
    )
    manifest = optimized_runtime.manifest
    lossless = bool(manifest.get("package", {}).get("lossless", True))
    if lossless:
        assert official_frames == optimized_frames, "official and optimized frame tokens differ"
    return {
        "frame_count": len(optimized_frames),
        "tokens_match": official_frames == optimized_frames,
    }


def compare_pytorch_prefill(run_id: str, atol: float) -> dict[str, object]:
    optimized_runtime = OptimizedTtsRuntime(run_id, seed=1234)
    voice = optimized_runtime.list_builtin_voices()[0]
    text_sample = optimized_runtime.manifest["text_samples"][0]
    request_rows = optimized_runtime.build_voice_clone_request_rows(
        voice["prompt_audio_codes"],
        text_sample["text_token_ids"],
    )
    input_ids = torch.tensor([request_rows["inputIds"]], dtype=torch.long)
    attention_mask = torch.tensor(request_rows["attentionMask"], dtype=torch.bool)
    model = AutoModelForCausalLM.from_pretrained(
        str(DEFAULT_TTS_CHECKPOINT_DIR),
        trust_remote_code=True,
    )
    model.to(device="cpu", dtype=torch.float32)
    model._set_attention_implementation("eager")
    model.eval()
    with torch.no_grad():
        torch_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    pytorch_last = torch_outputs.global_hidden_states[:, -1, :].detach().cpu().numpy()
    if not np.isfinite(pytorch_last).all():
        return {
            "pytorch_prefill_status": "skipped_nonfinite_official_cpu_forward",
            "pytorch_prefill_finite": False,
        }
    optimized_outputs = optimized_runtime.sessions["prefill"].run(
        None,
        {
            "input_ids": input_ids.numpy().astype(np.int32, copy=False),
            "attention_mask": attention_mask.numpy().astype(np.int32, copy=False),
        },
    )
    optimized_named = dict(
        zip([output.name for output in optimized_runtime.sessions["prefill"].get_outputs()], optimized_outputs, strict=True)
    )
    prefill_max, prefill_mean = _max_mean_abs(pytorch_last, optimized_named["global_hidden_last"])
    return {
        "pytorch_prefill_status": "compared" if prefill_max <= atol else "reported_mismatch",
        "pytorch_prefill_finite": True,
        "pytorch_prefill_within_tolerance": prefill_max <= atol,
        "pytorch_prefill_max_abs_error": prefill_max,
        "pytorch_prefill_mean_abs_error": prefill_mean,
    }


def run_alignment(run_id: str, max_new_frames: int, atol: float, skip_pytorch: bool = False) -> dict[str, object]:
    paths = assert_export_layout(run_id)
    assert_last_hidden_outputs(paths)
    assert_shared_external_data(paths)
    hidden_report = compare_prefill_and_decode_last_hidden(paths, atol)
    generation_report = compare_short_generation(run_id, max_new_frames)
    pytorch_report = {} if skip_pytorch else compare_pytorch_prefill(run_id, atol)
    report = {
        "run_id": run_id,
        "prefill_output": "global_hidden_last",
        "decode_output": "global_hidden_last",
        "external_data_shared": "global_data" in paths,
        **hidden_report,
        **generation_report,
        **pytorch_report,
    }
    report_path = paths["run_dir"] / "alignment_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    report = run_alignment(args.run_id, args.max_new_frames, args.atol, skip_pytorch=args.skip_pytorch)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
