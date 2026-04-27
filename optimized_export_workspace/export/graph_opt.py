from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def _dims(value_info: onnx.ValueInfoProto) -> list[int | str]:
    return [
        dim.dim_param if dim.dim_param else int(dim.dim_value)
        for dim in value_info.type.tensor_type.shape.dim
    ]


def load_output_shapes(model_path: str | Path) -> dict[str, list[int | str]]:
    model = onnx.load(str(model_path), load_external_data=False)
    return {output.name: _dims(output) for output in model.graph.output}


def external_data_locations(model_path: str | Path) -> set[str]:
    model = onnx.load(str(model_path), load_external_data=False)
    locations: set[str] = set()
    for initializer in model.graph.initializer:
        for entry in initializer.external_data:
            if entry.key == "location":
                locations.add(entry.value)
    return locations


def graph_stats(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path)
    model = onnx.load(str(path), load_external_data=False)
    op_counts = Counter(node.op_type for node in model.graph.node)
    return {
        "path": str(path),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "size_bytes": path.stat().st_size,
        "external_data_locations": sorted(external_data_locations(path)),
        "op_counts": dict(sorted(op_counts.items())),
    }


def export_ort_optimized_model(model_path: str | Path, *, thread_count: int = 1) -> dict[str, Any]:
    path = Path(model_path)
    temp_path = path.with_name(path.stem + ".ort_tmp.onnx")
    if temp_path.exists():
        temp_path.unlink()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = str(temp_path)
    options.intra_op_num_threads = max(1, int(thread_count))
    options.inter_op_num_threads = 1
    before = graph_stats(path)
    try:
        ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        if not temp_path.is_file():
            raise RuntimeError(f"ORT did not write optimized model: {temp_path}")
        shutil.move(str(temp_path), str(path))
        onnx.checker.check_model(str(path))
        after = graph_stats(path)
        return {
            "enabled": True,
            "status": "success",
            "before_nodes": before["nodes"],
            "after_nodes": after["nodes"],
            "before_size_bytes": before["size_bytes"],
            "after_size_bytes": after["size_bytes"],
        }
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        return {
            "enabled": True,
            "status": "failed",
            "error": str(exc),
            "before_nodes": before["nodes"],
            "after_nodes": before["nodes"],
            "before_size_bytes": before["size_bytes"],
            "after_size_bytes": before["size_bytes"],
        }


def simplify_model_to_external_data(
    source_path: str | Path,
    target_path: str | Path,
    *,
    external_data_name: str,
) -> dict[str, Any]:
    source = Path(source_path)
    target = Path(target_path)
    before = graph_stats(source)
    try:
        from onnxsim import simplify
    except ImportError as exc:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": f"onnxsim unavailable: {exc}",
            "before_nodes": before["nodes"],
            "after_nodes": before["nodes"],
        }
    try:
        model = onnx.load(str(source), load_external_data=True)
        simplified, ok = simplify(model, perform_optimization=True, skip_fuse_bn=True)
        if not ok:
            raise RuntimeError("onnxsim returned check=False")
        onnx.save_model(
            simplified,
            str(target),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_name,
            size_threshold=1024,
        )
        onnx.checker.check_model(str(target))
        after = graph_stats(target)
        return {
            "enabled": True,
            "status": "success",
            "before_nodes": before["nodes"],
            "after_nodes": after["nodes"],
            "before_size_bytes": before["size_bytes"],
            "after_size_bytes": after["size_bytes"],
        }
    except Exception as exc:
        return {
            "enabled": True,
            "status": "failed",
            "error": str(exc),
            "before_nodes": before["nodes"],
            "after_nodes": before["nodes"],
            "before_size_bytes": before["size_bytes"],
            "after_size_bytes": before["size_bytes"],
        }


def save_with_last_hidden_output(
    source_path: str | Path,
    target_path: str | Path,
    *,
    source_output_name: str = "global_hidden",
    target_output_name: str = "global_hidden_last",
) -> Path:
    source = Path(source_path)
    target = Path(target_path)
    model = onnx.load(str(source), load_external_data=False)
    graph = model.graph

    output_by_name = {output.name: output for output in graph.output}
    if source_output_name not in output_by_name:
        raise ValueError(f"{source_output_name!r} is not an output of {source}")
    source_output = output_by_name[source_output_name]
    source_dims = _dims(source_output)
    if len(source_dims) != 3:
        raise ValueError(f"expected {source_output_name} rank 3, got {source_dims}")
    batch_dim, _seq_dim, hidden_dim = source_dims

    starts_name = f"{target_output_name}_starts"
    ends_name = f"{target_output_name}_ends"
    axes_name = f"{target_output_name}_axes"
    steps_name = f"{target_output_name}_steps"
    sliced_name = f"{target_output_name}_rank3"

    graph.node.extend(
        [
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=[starts_name],
                value=helper.make_tensor(starts_name, TensorProto.INT64, [1], [-1]),
                name=f"{target_output_name}_starts_const",
            ),
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=[ends_name],
                value=helper.make_tensor(ends_name, TensorProto.INT64, [1], [9223372036854775807]),
                name=f"{target_output_name}_ends_const",
            ),
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=[axes_name],
                value=helper.make_tensor(axes_name, TensorProto.INT64, [1], [1]),
                name=f"{target_output_name}_axes_const",
            ),
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=[steps_name],
                value=helper.make_tensor(steps_name, TensorProto.INT64, [1], [1]),
                name=f"{target_output_name}_steps_const",
            ),
            helper.make_node(
                "Slice",
                inputs=[source_output_name, starts_name, ends_name, axes_name, steps_name],
                outputs=[sliced_name],
                name=f"{target_output_name}_slice_last_seq",
            ),
            helper.make_node(
                "Squeeze",
                inputs=[sliced_name, axes_name],
                outputs=[target_output_name],
                name=f"{target_output_name}_squeeze_seq",
            ),
        ]
    )

    new_outputs = [
        output for output in graph.output
        if output.name != source_output_name
    ]
    last_hidden_info = helper.make_tensor_value_info(
        target_output_name,
        source_output.type.tensor_type.elem_type,
        [batch_dim, hidden_dim],
    )
    graph.ClearField("output")
    graph.output.extend([last_hidden_info, *new_outputs])

    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(target), save_as_external_data=True, all_tensors_to_one_file=False)
    onnx.checker.check_model(str(target))
    return target
