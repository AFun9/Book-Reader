# Optimized ONNX Export Design

## Goal

Build a dedicated optimized ONNX export and inference workspace for MOSS-TTS-Nano, separate from the existing official ONNX runtime compatibility path. The optimized path should export models from the local official PyTorch checkpoints, run dedicated inference, and verify outputs against the official PyTorch and official ONNX assets.

## Workspace

All new implementation, tests, logs, exported models, reports, and generated audio live under:

```text
optimized_export_workspace/
```

The workspace owns these subdirectories:

- `export/`: export modules, graph optimization utilities, and command-line entry points.
- `runtime/`: optimized runtime code for the new model interface.
- `tests/`: alignment and smoke tests for the optimized export path.
- `runs/<run_id>/`: generated artifacts for each export or validation run.

Each `runs/<run_id>/` directory contains the optimized TTS model folder, copied or referenced codec assets, logs, JSON reports, and audio generated from that exact model build. This keeps logs and audio traceable to the exported model.

## Model Export Scope

The first optimized export covers the TTS autoregressive model. The existing local official codec ONNX assets are reused for encode/decode unless a later step shows a codec-side optimization is necessary.

The optimized TTS package contains:

- `moss_tts_prefill_last.onnx`: prefill graph whose global hidden output is only the final valid token hidden state with shape `[batch, hidden]`.
- `moss_tts_decode_step_last.onnx`: decode graph with the same final-hidden style output for consistency.
- `moss_tts_local_fixed_sampled_frame.onnx`: whole-frame fixed-parameter sampled local graph for the default sampling path.
- `moss_tts_local_greedy_frame.onnx` if a greedy path is needed for deterministic token-level validation.
- Shared external data files for global and local weights, without duplicating identical initializers across graphs.
- `optimized_manifest.json`: dedicated manifest for the optimized runtime, not constrained by `browser_poc_manifest.json`.

## Prefill Optimization

The official prefill graph returns `global_hidden` with shape `[batch, prefill_seq, hidden]`, while the runtime only consumes the last valid token hidden state. The optimized prefill graph returns `global_hidden_last` with shape `[batch, hidden]`.

For unpadded single-sample requests, this is equivalent to slicing `global_hidden[:, -1, :]`. If batched left-padding or mixed valid lengths are introduced, the graph must gather by `attention_mask.sum(axis=1) - 1` so each batch item uses its own final valid token.

The optimized runtime consumes the last-hidden output directly and does not include compatibility logic for the official full-sequence output.

## Graph Sharing And Size

The official ONNX package already stores TTS global weights in `moss_tts_global_shared.data` and local weights in `moss_tts_local_shared.data`. The optimized exporter should preserve this property and verify it in tests by checking initializer external-data locations.

Additional graph-size optimizations are allowed when they do not change output parity:

- Remove unused graph outputs.
- Run ONNX shape inference and model checking after graph edits.
- Use ONNX Runtime graph optimization during validation.
- Avoid generating duplicate model folders for the same run.

## Runtime

The optimized runtime is a new implementation under `optimized_export_workspace/runtime/`. It reads `optimized_manifest.json`, loads the optimized ONNX files, and follows the same high-level synthesis flow:

1. Build voice-clone request rows.
2. Run optimized prefill and keep only `global_hidden_last`.
3. Generate frames using the optimized local frame graph.
4. Run optimized or official-style decode step to update global hidden and KV cache.
5. Decode generated audio with the official codec ONNX assets.
6. Save audio under the same run directory as the model and report.

The runtime can borrow small pure-Python helpers from the repository where useful, but the optimized interface should remain explicit and independent.

## Validation

Validation is required before calling an export successful.

Primary checks:

- Official PyTorch model vs optimized ONNX for prefill last hidden.
- Official ONNX model vs optimized ONNX for prefill last hidden.
- Decode-step last hidden and KV-cache outputs match the official ONNX graph after slicing.
- Greedy generation frame tokens match across official and optimized runtimes.
- Generated WAV files are saved for official PyTorch, official ONNX, and optimized ONNX under the same run folder when practical.

Tolerance targets:

- Exact token match for greedy generation.
- Hidden-state max absolute error should be near float32 ONNX export noise; failures report max absolute error and mean absolute error.
- Sampling validation uses fixed RNG streams and records the random numbers used.

## Commands

The initial command surface should be:

```bash
conda run -n tts python optimized_export_workspace/export/export_optimized_tts.py --run-id <id>
conda run -n tts python optimized_export_workspace/tests/test_alignment.py --run-id <id>
conda run -n tts python optimized_export_workspace/runtime/infer_optimized.py --run-id <id> --text "..."
```

Scripts default to local paths already present in the repository:

- PyTorch TTS checkpoint: `MOSS-TTS-Nano-100M`
- official TTS ONNX: `MOSS-TTS-Nano-100M-ONNX`
- official codec PyTorch checkpoint: `MOSS-Audio-Tokenizer-Nano`
- official codec ONNX: `MOSS-Audio-Tokenizer-Nano-ONNX`

## Non-Goals

- Do not preserve compatibility with the existing `OnnxTtsRuntime` interface.
- Do not modify official model directories in place.
- Do not optimize model quality or sampling defaults.
- Do not require network downloads.

