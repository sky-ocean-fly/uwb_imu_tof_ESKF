#!/usr/bin/env python3
"""Build one-row-per-static-pose data while retaining a row-level audit CSV."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCE = PROJECT_ROOT / "src"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from uwb_bias_learning.datasets import (  # noqa: E402
    AGGREGATE_COLUMNS,
    ANNOTATION_COLUMNS,
    AggregationConfig,
    DatasetError,
    aggregate_pose_rows,
    assert_distinct_new_outputs,
    load_config_mapping,
    ordered_fieldnames,
    read_csv_rows,
    write_csv_exclusive,
    write_json_exclusive,
)


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="raw/enriched CSV")
    parser.add_argument(
        "--output", required=True, type=Path, help="new accepted pose-level CSV"
    )
    parser.add_argument(
        "--rejected-output",
        required=True,
        type=Path,
        help="new rejected-pose CSV used to plan recollection",
    )
    parser.add_argument(
        "--annotated-output",
        required=True,
        type=Path,
        help="new row-level CSV retaining base rejection and outlier reasons",
    )
    parser.add_argument(
        "--summary-report", type=Path, help="optional new JSON build summary"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "preprocessing.yaml",
        help="preprocessing YAML/JSON",
    )
    parser.add_argument("--min-valid-ranges", type=int)
    parser.add_argument("--source-id", help="confirmed directed UWB source ID")
    parser.add_argument("--peer-id", help="confirmed directed UWB peer ID")
    parser.add_argument(
        "--anchor-id", help="legacy alias for --source-id (no physical anchor role inferred)"
    )
    parser.add_argument(
        "--node-id", help="legacy alias for --peer-id (no physical node role inferred)"
    )
    parser.add_argument(
        "--disable-outlier-filter",
        action="store_true",
        help="explicitly retain every row that passes the base quality gates",
    )
    parser.add_argument(
        "--allow-unconfigured-id-filter",
        action="store_true",
        help="process-only/synthetic scaffolding; forbidden for a formal A-B dataset",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] = None) -> int:
    args = parse_args(argv)
    try:
        outputs = [args.output, args.rejected_output, args.annotated_output]
        if args.summary_report is not None:
            outputs.append(args.summary_report)
        assert_distinct_new_outputs([args.input, args.config], outputs)

        mapping = load_config_mapping(args.config)
        config = AggregationConfig.from_mapping(mapping)
        if args.min_valid_ranges is not None:
            config = replace(config, min_valid_ranges=args.min_valid_ranges)
        if (args.source_id is not None or args.peer_id is not None) and (
            args.anchor_id is not None or args.node_id is not None
        ):
            raise DatasetError("use source/peer options or legacy aliases, not both")
        source_id = args.source_id if args.source_id is not None else args.anchor_id
        peer_id = args.peer_id if args.peer_id is not None else args.node_id
        if (source_id is None) != (peer_id is None):
            raise DatasetError("--source-id and --peer-id must be supplied together")
        if source_id is not None:
            config = replace(
                config,
                expected_anchor_id=str(source_id),
                expected_node_id=str(peer_id),
            )
        if config.expected_anchor_id is None and not args.allow_unconfigured_id_filter:
            raise DatasetError(
                "the confirmed directed UWB pair is not configured; pass --source-id and "
                "--peer-id after Gate 1. Use --allow-unconfigured-id-filter only for "
                "explicitly synthetic/process-only data."
            )
        if args.disable_outlier_filter:
            config = replace(config, outlier=replace(config.outlier, enabled=False))

        rows, input_fields = read_csv_rows(args.input)
        result = aggregate_pose_rows(rows, config=config)
        result.summary["source_csv"] = str(args.input)
        result.summary["config_file"] = str(args.config)
        result.summary["id_filter_active"] = config.expected_anchor_id is not None
        if config.expected_anchor_id is None:
            result.summary["warning"] = (
                "ID filtering was explicitly disabled for scaffolding; output is not a "
                "formal single-pair training dataset"
            )

        write_csv_exclusive(args.output, result.pose_rows, AGGREGATE_COLUMNS)
        write_csv_exclusive(
            args.rejected_output, result.rejected_pose_rows, AGGREGATE_COLUMNS
        )
        annotated_fields = ordered_fieldnames(
            result.annotated_rows, tuple(input_fields) + ANNOTATION_COLUMNS
        )
        write_csv_exclusive(
            args.annotated_output, result.annotated_rows, annotated_fields
        )
        if args.summary_report is not None:
            write_json_exclusive(args.summary_report, result.summary)

        print(
            "pose build: accepted={accepted}, rejected={rejected}, outliers={outliers}".format(
                accepted=len(result.pose_rows),
                rejected=len(result.rejected_pose_rows),
                outliers=result.summary["outlier_count"],
            )
        )
        print("accepted dataset: {}".format(args.output))
        if not result.pose_rows:
            print(
                "no pose passed all quality gates; inspect {} and {}".format(
                    args.rejected_output, args.annotated_output
                ),
                file=sys.stderr,
            )
            return 2
        return 0
    except (DatasetError, OSError, ValueError) as exc:
        print("build_pose_dataset: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
