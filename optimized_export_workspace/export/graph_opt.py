from __future__ import annotations

from pathlib import Path

import onnx
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
