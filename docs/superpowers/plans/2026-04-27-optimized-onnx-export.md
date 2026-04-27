# Optimized ONNX Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dedicated optimized ONNX export workspace for MOSS-TTS-Nano with last-hidden prefill/decode graphs, dedicated runtime, and alignment tests.

**Architecture:** The exporter creates a run folder under `optimized_export_workspace/runs/<run_id>/`, copies metadata/tokenizer assets, rewrites official ONNX graph outputs to last-hidden tensors, and preserves external shared data references. The optimized runtime reads `optimized_manifest.json` and uses only the optimized interface instead of adapting to the existing official runtime.

**Tech Stack:** Python 3, ONNX, ONNX Runtime, NumPy, SentencePiece, Torch/Torchaudio for audio loading helpers, existing repository pure-Python prompt/runtime helpers.

---

### Task 1: Workspace Core

**Files:**
- Create: `optimized_export_workspace/__init__.py`
- Create: `optimized_export_workspace/common.py`
- Create: `optimized_export_workspace/export/__init__.py`
- Create: `optimized_export_workspace/runtime/__init__.py`
- Create: `optimized_export_workspace/tests/__init__.py`

- [ ] **Step 1: Create path helpers**

Implement `common.py` with repo-root discovery, default model paths, run directory creation, JSON helpers, and timestamp run-id generation.

- [ ] **Step 2: Verify import**

Run: `conda run -n tts python -c "from optimized_export_workspace.common import REPO_ROOT; print(REPO_ROOT.name)"`

Expected: prints `MOSS-TTS-Nano`.

### Task 2: Optimized ONNX Graph Export

**Files:**
- Create: `optimized_export_workspace/export/graph_opt.py`
- Create: `optimized_export_workspace/export/export_optimized_tts.py`

- [ ] **Step 1: Add graph rewrite helpers**

Implement a helper that loads an official ONNX graph without embedding external data, slices `global_hidden[:, -1, :]`, replaces that graph output with `global_hidden_last`, checks the model, and saves it to the run model directory while keeping external data locations shared.

- [ ] **Step 2: Add export CLI**

Implement `export_optimized_tts.py` to create `runs/<run_id>/model/tts`, copy tokenizer and JSON metadata, copy shared data files, export `moss_tts_prefill_last.onnx` and `moss_tts_decode_step_last.onnx`, copy local frame graphs, write `optimized_manifest.json`, and write `export_report.json`.

- [ ] **Step 3: Run exporter**

Run: `conda run -n tts python optimized_export_workspace/export/export_optimized_tts.py --run-id smoke`

Expected: `optimized_export_workspace/runs/smoke/model/tts/optimized_manifest.json` exists.

### Task 3: Dedicated Optimized Runtime

**Files:**
- Create: `optimized_export_workspace/runtime/optimized_runtime.py`
- Create: `optimized_export_workspace/runtime/infer_optimized.py`

- [ ] **Step 1: Implement runtime**

Implement a runtime that loads `optimized_manifest.json`, uses optimized prefill/decode sessions, reuses official codec sessions, builds voice-clone rows from manifest templates, runs frame generation with optimized local fixed sampled or copied local cached/decoder graph, and writes generated WAV files inside `runs/<run_id>/audio/`.

- [ ] **Step 2: Implement CLI**

Implement `infer_optimized.py` with `--run-id`, `--text`, `--prompt-audio-path`, `--voice`, `--output-name`, `--seed`, `--max-new-frames`, and `--sample-mode`.

- [ ] **Step 3: Run short inference**

Run: `conda run -n tts python optimized_export_workspace/runtime/infer_optimized.py --run-id smoke --text "欢迎使用优化导出模型。" --max-new-frames 8 --output-name smoke.wav`

Expected: `optimized_export_workspace/runs/smoke/audio/smoke.wav` exists.

### Task 4: Alignment Tests And Reports

**Files:**
- Create: `optimized_export_workspace/tests/test_alignment.py`

- [ ] **Step 1: Implement alignment tests**

Implement tests that compare official ONNX prefill/decode sliced hidden outputs against optimized prefill/decode last-hidden outputs, verify external data sharing, compare greedy frame tokens for a short generation, and save `alignment_report.json`.

- [ ] **Step 2: Run alignment**

Run: `conda run -n tts python optimized_export_workspace/tests/test_alignment.py --run-id smoke --max-new-frames 8`

Expected: report shows hidden errors and greedy token comparison status.

### Task 5: Final Verification

**Files:**
- Modify only if needed based on failures: files under `optimized_export_workspace/`

- [ ] **Step 1: Run exporter, alignment, and inference end-to-end**

Run the three smoke commands from Tasks 2-4.

- [ ] **Step 2: Check generated artifact layout**

Run: `find optimized_export_workspace/runs/smoke -maxdepth 3 -type f | sort`

Expected: model files, logs/reports, and generated audio are grouped under the same `smoke` run directory.

