#!/usr/bin/env python3
"""Audit a raw UWB CSV and write a non-overwriting JSON quality report."""

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
    ANNOTATION_COLUMNS,
    AggregationConfig,
    compare_coverage_to_pose_domain,
    DatasetError,
    assert_distinct_new_outputs,
    load_config_mapping,
    ordered_fieldnames,
    read_csv_rows,
    validate_raw_rows,
    write_csv_exclusive,
    write_json_exclusive,
)


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="raw UTF-8 CSV")
    parser.add_argument(
        "--report", required=True, type=Path, help="new JSON report path"
    )
    parser.add_argument(
        "--annotated-output",
        type=Path,
        help="optional new CSV containing derived geometry and audit status",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "preprocessing.yaml",
        help="preprocessing YAML/JSON",
    )
    parser.add_argument(
        "--pose-domain-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pose_domain.yaml",
        help="pose-domain YAML/JSON used for an explicit coverage comparison",
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
        "--allow-incomplete-poses",
        action="store_true",
        help="report poses below min_valid_ranges without making the audit fail",
    )
    parser.add_argument(
        "--allow-derived-schema",
        action="store_true",
        help="accept a derived table without every phase-3 raw column",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] = None) -> int:
    args = parse_args(argv)
    try:
        output_paths = [args.report]
        if args.annotated_output is not None:
            output_paths.append(args.annotated_output)
        assert_distinct_new_outputs(
            [args.input, args.config, args.pose_domain_config],
            output_paths,
        )
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

        rows, input_fields = read_csv_rows(args.input)
        result = validate_raw_rows(
            rows,
            config=config,
            require_complete_raw_schema=not args.allow_derived_schema,
            fail_incomplete_poses=not args.allow_incomplete_poses,
        )
        pose_domain = load_config_mapping(args.pose_domain_config)
        domain_comparison = compare_coverage_to_pose_domain(
            result.report.get("coverage", {}), pose_domain
        )
        result.report["pose_domain_file"] = str(args.pose_domain_config)
        result.report["pose_domain_comparison"] = domain_comparison
        if domain_comparison["violations"]:
            result.severe_errors.extend(
                "pose domain violation: {}".format(message)
                for message in domain_comparison["violations"]
            )
            result.report["severe_errors"] = result.severe_errors
            result.report["passed"] = False
        result.report["source_csv"] = str(args.input)
        result.report["config_file"] = str(args.config)
        write_json_exclusive(args.report, result.report)
        if args.annotated_output is not None:
            fields = ordered_fieldnames(
                result.annotated_rows, tuple(input_fields) + ANNOTATION_COLUMNS
            )
            write_csv_exclusive(args.annotated_output, result.annotated_rows, fields)

        print(
            "raw audit: rows={rows}, valid={valid}, invalid={invalid}, poses={poses}".format(
                rows=result.report["row_count"],
                valid=result.report["base_valid_count"],
                invalid=result.report["base_invalid_count"],
                poses=result.report["pose_count"],
            )
        )
        print("report: {}".format(args.report))
        if result.passed:
            print("status: PASS")
            return 0
        print("status: FAIL", file=sys.stderr)
        for error in result.severe_errors:
            print("- {}".format(error), file=sys.stderr)
        return 2
    except (DatasetError, OSError, ValueError) as exc:
        print("validate_raw_dataset: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
