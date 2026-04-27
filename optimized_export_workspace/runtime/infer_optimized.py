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

from optimized_export_workspace.runtime.optimized_runtime import OptimizedTtsRuntime
from optimized_export_workspace.common import write_json


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dedicated optimized MOSS-TTS-Nano ONNX inference.")
    parser.add_argument("--run-id", required=True, help="Run id under optimized_export_workspace/runs.")
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    parser.add_argument("--voice", default="Junhao", help="Built-in voice preset when no prompt audio is provided.")
    parser.add_argument("--prompt-audio-path", default=None, help="Reference audio path for voice cloning.")
    parser.add_argument("--output-name", default="optimized_output.wav", help="Output WAV filename under the run audio dir.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for fixed sampled frame generation.")
    parser.add_argument("--cpu-threads", type=int, default=4, help="ONNX Runtime intra-op thread count.")
    parser.add_argument("--max-new-frames", type=int, default=None, help="Maximum generated audio frames.")
    parser.add_argument("--sample-mode", default="fixed", choices=("fixed",), help="Optimized runtime currently exports fixed sampling.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    runtime = OptimizedTtsRuntime(args.run_id, thread_count=args.cpu_threads, seed=args.seed)
    result = runtime.synthesize(
        text=args.text,
        voice=args.voice,
        prompt_audio_path=args.prompt_audio_path,
        output_name=args.output_name,
        max_new_frames=args.max_new_frames,
    )
    report_path = runtime.run_dir / "logs" / f"infer_{Path(args.output_name).stem}.json"
    write_json(
        report_path,
        {
            "event": "optimized_tts_inference",
            "run_id": args.run_id,
            "text": args.text,
            "voice": args.voice,
            "prompt_audio_path": args.prompt_audio_path,
            "output_name": args.output_name,
            **result,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
