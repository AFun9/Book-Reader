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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check graph optimization reports for an export run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expect-ort", action="store_true")
    parser.add_argument("--expect-simplify", action="store_true")
    return parser.parse_args(argv)


def assert_graph_optimization_report(
    run_id: str,
    *,
    expect_ort: bool = False,
    expect_simplify: bool = False,
) -> dict[str, object]:
    run_dir = run_dir_for_id(run_id)
    report_path = run_dir / "graph_optimization_report.json"
    assert report_path.is_file(), f"missing graph optimization report: {report_path}"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    graphs = report["graphs"]
    assert {"prefill", "decode_step", "local_fixed_sampled_frame"} <= set(graphs)
    for graph_name, graph_report in graphs.items():
        assert graph_report["source_nodes"] > 0, graph_name
        assert graph_report["final_nodes"] > 0, graph_name
        assert graph_report["final_size_bytes"] > 0, graph_name
        assert graph_report["source_op_counts"], graph_name
        assert graph_report["final_op_counts"], graph_name
        if expect_ort:
            assert graph_report["ort"]["enabled"] is True, graph_name
            assert graph_report["ort"]["status"] == "success", graph_name
        if expect_simplify:
            assert graph_report["simplify"]["enabled"] is True, graph_name
            assert graph_report["simplify"]["status"] in {"success", "skipped"}, graph_name
    return report


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    report = assert_graph_optimization_report(
        args.run_id,
        expect_ort=args.expect_ort,
        expect_simplify=args.expect_simplify,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
