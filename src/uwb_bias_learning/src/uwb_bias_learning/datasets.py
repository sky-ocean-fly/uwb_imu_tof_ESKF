"""CSV data preparation for offline UWB bias learning.

This module deliberately uses :mod:`csv` from the Python standard library.  A
raw file is never modified in place: validation and aggregation return copied,
annotated rows which callers must write to a different path.

All quaternions exposed by this module use ``[qx, qy, qz, qw]`` order.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .transforms import average_quaternions, quaternion_to_matrix, relative_pose


POSE_KEY_FIELDS: Tuple[str, str] = ("session_id", "pose_id")
REL_TRANSLATION_FIELDS: Tuple[str, ...] = ("rel_tx", "rel_ty", "rel_tz")
REL_QUATERNION_FIELDS: Tuple[str, ...] = ("rel_qx", "rel_qy", "rel_qz", "rel_qw")
REL_MATRIX_FIELDS: Tuple[str, ...] = tuple(
    "rel_R{}{}".format(row, column) for row in range(3) for column in range(3)
)
POSE12_FIELDS: Tuple[str, ...] = REL_TRANSLATION_FIELDS + REL_MATRIX_FIELDS

A_POSITION_FIELDS: Tuple[str, ...] = ("A_px", "A_py", "A_pz")
A_QUATERNION_FIELDS: Tuple[str, ...] = ("A_qx", "A_qy", "A_qz", "A_qw")
B_POSITION_FIELDS: Tuple[str, ...] = ("B_px", "B_py", "B_pz")
B_QUATERNION_FIELDS: Tuple[str, ...] = ("B_qx", "B_qy", "B_qz", "B_qw")

RAW_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "data_origin",
    "is_synthetic",
    "ingestion_path",
    "session_id",
    "environment_id",
    "pose_id",
    "attempt_id",
    "sample_index",
    "capture_state",
    "uwb_timestamp",
    "uwb_receive_timestamp",
    "uwb_timestamp_source",
    "range_m",
    "source_id",
    "peer_id",
    "anchor_id",
    "node_id",
    "fp_rssi",
    "rx_rssi",
    "cir_quality",
    "mocap_a_timestamp_before",
    "mocap_a_timestamp_after",
    "mocap_b_timestamp_before",
    "mocap_b_timestamp_after",
    "sync_gap_a_s",
    "sync_gap_b_s",
    *A_POSITION_FIELDS,
    *A_QUATERNION_FIELDS,
    *B_POSITION_FIELDS,
    *B_QUATERNION_FIELDS,
    "is_los",
    "is_stable",
    "is_mocap_valid",
    "is_sync_valid",
    "is_range_valid",
    "invalid_reason",
    "operator_note",
)

AGGREGATE_COLUMNS: Tuple[str, ...] = (
    "session_id",
    "pose_id",
    "environment_id",
    "data_origin",
    "is_synthetic",
    *REL_TRANSLATION_FIELDS,
    *REL_QUATERNION_FIELDS,
    *REL_MATRIX_FIELDS,
    "distance_gt_mean",
    "distance_gt_std",
    "range_mean_raw",
    "range_mean_filtered",
    "range_median",
    "range_median_raw",
    "range_median_filtered",
    "range_std_raw",
    "range_mad_raw",
    "range_q05_raw",
    "range_q25_raw",
    "range_q75_raw",
    "range_q95_raw",
    "bias_mean_raw",
    "bias_mean",
    "bias_median_raw",
    "bias_median_filtered",
    "bias_std_raw",
    "bias_mad_raw",
    "bias_q05_raw",
    "bias_q25_raw",
    "bias_q75_raw",
    "bias_q95_raw",
    "bias_std",
    "bias_sem",
    "row_count_total",
    "base_rejected_count",
    "range_count_raw",
    "range_count_valid",
    "outlier_count",
    "position_stability_m",
    "rotation_stability_deg",
    "mean_fp_rssi",
    "mean_rx_rssi",
    "outlier_method",
    "outlier_center",
    "outlier_mad",
    "outlier_cutoff",
    "quality_status",
    "rejection_reason",
)

ANNOTATION_COLUMNS: Tuple[str, ...] = (
    *REL_TRANSLATION_FIELDS,
    *REL_QUATERNION_FIELDS,
    *REL_MATRIX_FIELDS,
    "distance_gt_m",
    "range_error_m",
    "processing_status",
    "processing_invalid_reason",
    "is_outlier",
    "outlier_reason",
)


class DatasetError(ValueError):
    """Base exception for malformed or unsafe dataset operations."""


class DatasetSchemaError(DatasetError):
    """Raised when required columns or values are absent."""


class OutputExistsError(DatasetError):
    """Raised when an output would overwrite an existing file."""


@dataclass(frozen=True)
class OutlierConfig:
    """Configuration for within-pose communication-outlier removal."""

    enabled: bool = True
    method: str = "hampel_mad"
    field: str = "range_error_m"
    threshold: float = 3.0
    mad_scale: float = 1.4826
    min_samples: int = 4
    minimum_absolute_deviation_m: float = 0.0

    def __post_init__(self) -> None:
        if self.method not in ("hampel_mad", "none"):
            raise DatasetError("outlier.method must be 'hampel_mad' or 'none'")
        if self.field not in ("range_error_m", "range_m"):
            raise DatasetError("outlier.field must be 'range_error_m' or 'range_m'")
        if self.threshold <= 0.0 or not math.isfinite(self.threshold):
            raise DatasetError("outlier.threshold must be finite and > 0")
        if self.mad_scale <= 0.0 or not math.isfinite(self.mad_scale):
            raise DatasetError("outlier.mad_scale must be finite and > 0")
        if self.min_samples < 1:
            raise DatasetError("outlier.min_samples must be >= 1")
        if self.minimum_absolute_deviation_m < 0.0:
            raise DatasetError("outlier.minimum_absolute_deviation_m must be >= 0")

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]]) -> "OutlierConfig":
        if not values:
            return cls()
        return cls(
            enabled=_mapping_bool(values, "enabled", True),
            method=str(values.get("method", "hampel_mad")).strip().lower(),
            field=str(values.get("field", "range_error_m")).strip(),
            threshold=float(
                values.get(
                    "threshold",
                    values.get("threshold_mad", values.get("n_sigma", 3.0)),
                )
            ),
            mad_scale=float(values.get("mad_scale", 1.4826)),
            min_samples=int(values.get("min_samples", 4)),
            minimum_absolute_deviation_m=float(
                values.get(
                    "minimum_absolute_deviation_m",
                    values.get("minimum_scale_m", 0.0),
                )
            ),
        )


@dataclass(frozen=True)
class AggregationConfig:
    """Quality gates and aggregation settings.

    IDs are optional while an installation is being scaffolded.  When either
    ID is configured both are required and rows are checked against that exact
    directed pair; no direction is guessed.
    """

    min_valid_ranges: int = 180
    expected_anchor_id: Optional[str] = None
    expected_node_id: Optional[str] = None
    allow_reverse_ids: bool = False
    require_los: bool = True
    range_min_m: float = 0.0
    range_max_m: Optional[float] = None
    geometry_distance_tolerance_m: float = 1.0e-4
    rotation_orthogonality_tolerance: float = 1.0e-6
    rotation_determinant_tolerance: float = 1.0e-6
    outlier: OutlierConfig = field(default_factory=OutlierConfig)

    def __post_init__(self) -> None:
        if self.min_valid_ranges < 1:
            raise DatasetError("min_valid_ranges must be >= 1")
        exactly_one_id = (self.expected_anchor_id is None) != (
            self.expected_node_id is None
        )
        if exactly_one_id:
            raise DatasetError(
                "expected_anchor_id and expected_node_id must be configured together"
            )
        if self.allow_reverse_ids:
            raise DatasetError(
                "reverse ID matching is not permitted; configure the exact source_id/peer_id direction"
            )
        if not math.isfinite(self.range_min_m):
            raise DatasetError("range_min_m must be finite")
        if self.range_max_m is not None:
            if not math.isfinite(self.range_max_m):
                raise DatasetError("range_max_m must be finite when configured")
            if self.range_max_m <= self.range_min_m:
                raise DatasetError("range_max_m must be greater than range_min_m")
        if self.geometry_distance_tolerance_m < 0.0:
            raise DatasetError("geometry_distance_tolerance_m must be >= 0")
        if self.rotation_orthogonality_tolerance < 0.0:
            raise DatasetError("rotation_orthogonality_tolerance must be >= 0")
        if self.rotation_determinant_tolerance < 0.0:
            raise DatasetError("rotation_determinant_tolerance must be >= 0")

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]]) -> "AggregationConfig":
        if not values:
            return cls()
        root = values
        for key in ("preprocessing", "aggregation"):
            nested = root.get(key)
            if isinstance(nested, Mapping):
                root = nested
                break

        pair = root.get("uwb_pair", {})
        if not isinstance(pair, Mapping):
            pair = {}
        anchor = root.get(
            "expected_source_id",
            root.get(
                "expected_anchor_id", pair.get("source_id", pair.get("anchor_id"))
            ),
        )
        node = root.get(
            "expected_peer_id",
            root.get("expected_node_id", pair.get("peer_id", pair.get("node_id"))),
        )
        anchor_text = None if anchor is None or str(anchor).strip() == "" else str(anchor)
        node_text = None if node is None or str(node).strip() == "" else str(node)

        outlier_values = root.get("outlier", root.get("outlier_filter", {}))
        if not isinstance(outlier_values, Mapping):
            raise DatasetError("outlier configuration must be a mapping")
        required_flags = root.get("required_flags", {})
        if not isinstance(required_flags, Mapping):
            required_flags = {}
        rotation_validity = root.get("rotation_validity", {})
        if not isinstance(rotation_validity, Mapping):
            raise DatasetError("rotation_validity configuration must be a mapping")
        maximum = root.get("range_max_m")
        return cls(
            min_valid_ranges=int(root.get("min_valid_ranges", 180)),
            expected_anchor_id=anchor_text,
            expected_node_id=node_text,
            allow_reverse_ids=_mapping_bool(root, "allow_reverse_ids", False),
            require_los=(
                _mapping_bool(required_flags, "is_los", True)
                if "require_los" not in root
                else _mapping_bool(root, "require_los", True)
            ),
            range_min_m=float(root.get("range_min_m", 0.0)),
            range_max_m=None if maximum is None else float(maximum),
            geometry_distance_tolerance_m=float(
                root.get("geometry_distance_tolerance_m", 1.0e-4)
            ),
            rotation_orthogonality_tolerance=float(
                rotation_validity.get("orthogonality_tolerance", 1.0e-6)
            ),
            rotation_determinant_tolerance=float(
                rotation_validity.get("determinant_tolerance", 1.0e-6)
            ),
            outlier=OutlierConfig.from_mapping(outlier_values),
        )

    @property
    def expected_source_id(self) -> Optional[str]:
        """Confirmed directed source ID (legacy storage name kept for API compatibility)."""

        return self.expected_anchor_id

    @property
    def expected_peer_id(self) -> Optional[str]:
        """Confirmed directed peer ID (legacy storage name kept for API compatibility)."""

        return self.expected_node_id


@dataclass
class AggregationResult:
    pose_rows: List[Dict[str, Any]]
    rejected_pose_rows: List[Dict[str, Any]]
    annotated_rows: List[Dict[str, Any]]
    summary: Dict[str, Any]


@dataclass
class RawValidationResult:
    annotated_rows: List[Dict[str, Any]]
    report: Dict[str, Any]
    severe_errors: List[str]

    @property
    def passed(self) -> bool:
        return not self.severe_errors


@dataclass
class SplitResult:
    train_rows: List[Dict[str, Any]]
    val_rows: List[Dict[str, Any]]
    test_rows: List[Dict[str, Any]]
    normalization: Dict[str, Any]
    manifest: Dict[str, Any]


def select_raw_test_rows(
    annotated_rows: Sequence[Mapping[str, Any]],
    test_pose_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Select valid, non-outlier measurements for the fixed test poses.

    The result is the row-level input required to evaluate corrected ranges.
    Selection is driven exclusively by the already-fixed pose split; no row is
    randomly sampled and no test statistic influences the split.
    """

    if not annotated_rows:
        raise DatasetError("annotated raw dataset is empty")
    test_keys = {
        (_text(row.get("session_id")), _text(row.get("pose_id")))
        for row in test_pose_rows
    }
    if not test_keys or any(not session or not pose for session, pose in test_keys):
        raise DatasetError("fixed test split contains an empty session_id or pose_id")

    required = (
        "session_id",
        "pose_id",
        "range_m",
        "distance_gt_m",
        "processing_status",
        "is_outlier",
        *POSE12_FIELDS,
    )
    headers = set().union(*(row.keys() for row in annotated_rows))
    missing = sorted(set(required) - headers)
    if missing:
        raise DatasetSchemaError(
            "annotated raw dataset lacks raw-test fields: {}".format(", ".join(missing))
        )

    selected: List[Dict[str, Any]] = []
    selected_keys: set = set()
    sample_keys: set = set()
    for row_index, source in enumerate(annotated_rows):
        key = (_text(source.get("session_id")), _text(source.get("pose_id")))
        if key not in test_keys:
            continue
        status = _text(source.get("processing_status")).lower()
        outlier = _optional_bool(source.get("is_outlier"))
        if status == "valid" and outlier is None:
            raise DatasetSchemaError(
                "annotated raw row {} has valid status but no explicit is_outlier boolean".format(
                    row_index
                )
            )
        if status != "valid":
            continue
        if outlier is not False:
            raise DatasetError(
                "annotated raw row {} is simultaneously valid and marked outlier".format(
                    row_index
                )
            )
        range_value = _required_finite_float(source.get("range_m"), "range_m", row_index)
        distance_value = _required_finite_float(
            source.get("distance_gt_m"), "distance_gt_m", row_index
        )
        if range_value <= 0.0 or distance_value < 0.0:
            raise DatasetError(
                "annotated raw row {} has an invalid range or geometric distance".format(
                    row_index
                )
            )
        geometry = _row_geometry(source, row_index)
        geometric_distance = math.sqrt(sum(value * value for value in geometry[0]))
        if abs(geometric_distance - distance_value) > 1.0e-4:
            raise DatasetError(
                "annotated raw row {} distance_gt_m disagrees with relative translation".format(
                    row_index
                )
            )
        sample_index = _text(source.get("sample_index"))
        if sample_index:
            attempt_id = _text(source.get("attempt_id"))
            sample_key = (
                (key[0], key[1], attempt_id, sample_index)
                if attempt_id
                else (key[0], key[1], sample_index)
            )
            if sample_key in sample_keys:
                raise DatasetError(
                    "duplicate test raw sample key: {}".format(
                        "/".join(sample_key)
                    )
                )
            sample_keys.add(sample_key)
        selected.append(dict(source))
        selected_keys.add(key)

    absent = sorted(test_keys - selected_keys)
    if absent:
        examples = ["{}/{}".format(*key) for key in absent[:5]]
        raise DatasetError(
            "annotated raw data has no valid non-outlier rows for {} fixed test poses; "
            "examples: {}".format(len(absent), ", ".join(examples))
        )
    return selected


def load_config_mapping(path: os.PathLike[str] | str) -> Dict[str, Any]:
    """Load a JSON or YAML mapping with an explicit error on missing YAML support."""

    config_path = Path(path)
    if not config_path.is_file():
        raise DatasetError("configuration file does not exist: {}".format(config_path))
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        parsed = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise DatasetError(
                "PyYAML is required to read configuration file {}".format(config_path)
            ) from exc
        parsed = yaml.safe_load(text)
    if parsed is None:
        return {}
    if not isinstance(parsed, Mapping):
        raise DatasetError("configuration root must be a mapping: {}".format(config_path))
    return dict(parsed)


def read_csv_rows(path: os.PathLike[str] | str) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read UTF-8 CSV using the standard library and retain header order."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise DatasetError("input CSV does not exist: {}".format(csv_path))
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise DatasetSchemaError("CSV has no header: {}".format(csv_path))
        fields = [str(name) for name in reader.fieldnames]
        if len(fields) != len(set(fields)):
            duplicates = sorted(name for name, count in Counter(fields).items() if count > 1)
            raise DatasetSchemaError(
                "CSV contains duplicate columns: {}".format(", ".join(duplicates))
            )
        rows = [dict(row) for row in reader]
    return rows, fields


def ordered_fieldnames(
    rows: Sequence[Mapping[str, Any]], preferred: Sequence[str] = ()
) -> List[str]:
    """Return deterministic field order, keeping preferred names first."""

    seen = set()
    result: List[str] = []
    for name in preferred:
        if name not in seen:
            seen.add(name)
            result.append(name)
    for row in rows:
        for name in row.keys():
            if name not in seen:
                seen.add(name)
                result.append(name)
    return result


def write_csv_exclusive(
    path: os.PathLike[str] | str,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    """Write CSV with mode ``x`` so an existing file is never overwritten."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or ordered_fieldnames(rows))
    if not fields:
        raise DatasetSchemaError("cannot write CSV without field names: {}".format(output))
    try:
        with output.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: _csv_value(row.get(name, "")) for name in fields})
    except FileExistsError as exc:
        raise OutputExistsError("refusing to overwrite existing file: {}".format(output)) from exc


def write_json_exclusive(path: os.PathLike[str] | str, value: Any) -> None:
    """Write human-readable JSON without overwriting an existing file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise OutputExistsError("refusing to overwrite existing file: {}".format(output)) from exc


def file_sha256(path: os.PathLike[str] | str) -> str:
    """Return the hexadecimal SHA-256 of a file without loading it all at once."""

    source = Path(path)
    if not source.is_file():
        raise DatasetError("cannot hash missing file: {}".format(source))
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def assert_distinct_new_outputs(
    input_paths: Sequence[os.PathLike[str] | str],
    output_paths: Sequence[os.PathLike[str] | str],
) -> None:
    """Reject aliases, duplicate targets, and existing targets before a script writes."""

    inputs = {_resolved_path(path) for path in input_paths if path is not None}
    outputs = [_resolved_path(path) for path in output_paths if path is not None]
    if len(outputs) != len(set(outputs)):
        raise DatasetError("output paths must be distinct")
    for output in outputs:
        if output in inputs:
            raise DatasetError("output must differ from every input: {}".format(output))
        if output.exists():
            raise OutputExistsError(
                "refusing to overwrite existing output: {}".format(output)
            )


def validate_raw_rows(
    rows: Sequence[Mapping[str, Any]],
    config: Optional[AggregationConfig] = None,
    require_complete_raw_schema: bool = True,
    fail_incomplete_poses: bool = True,
) -> RawValidationResult:
    """Validate and annotate raw measurement rows without changing the input rows."""

    cfg = (
        AggregationConfig.from_mapping(config)
        if isinstance(config, Mapping)
        else (config or AggregationConfig())
    )
    severe: List[str] = []
    if not rows:
        return RawValidationResult(
            annotated_rows=[],
            report={"passed": False, "row_count": 0, "severe_errors": ["input has no rows"]},
            severe_errors=["input has no rows"],
        )
    try:
        provenance = dataset_provenance(rows)
    except DatasetError as exc:
        provenance = "invalid_mixed"
        severe.append(str(exc))
    if provenance != "invalid_mixed":
        try:
            ingestion_path = canonical_raw_ingestion_path(rows, provenance)
        except DatasetError as exc:
            ingestion_path = "invalid"
            severe.append(str(exc))
    else:
        ingestion_path = "invalid"

    headers = set().union(*(row.keys() for row in rows))
    if require_complete_raw_schema:
        missing = sorted(set(RAW_REQUIRED_COLUMNS) - headers)
        if missing:
            severe.append("missing required raw columns: {}".format(", ".join(missing)))

    annotated, parsed, duplicate_counts = _annotate_base_rows(rows, cfg)
    reason_counts: Counter[str] = Counter()
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for index, row in enumerate(annotated):
        key = (_text(row.get("session_id")), _text(row.get("pose_id")))
        groups[key].append(index)
        reasons = _split_reasons(_text(row.get("processing_invalid_reason")))
        reason_counts.update(reasons)

    group_reports: List[Dict[str, Any]] = []
    insufficient: List[str] = []
    for key in sorted(groups):
        indices = groups[key]
        valid_indices = [index for index in indices if parsed[index] is not None]
        timestamps = [
            _optional_finite_float(annotated[index].get("uwb_timestamp"))
            for index in indices
        ]
        timestamp_stats = _timestamp_statistics(
            [value for value in timestamps if value is not None]
        )
        sync_a = _finite_values(annotated[index].get("sync_gap_a_s") for index in indices)
        sync_b = _finite_values(annotated[index].get("sync_gap_b_s") for index in indices)
        ranges = [float(parsed[index]["range_m"]) for index in valid_indices]
        distances = [float(parsed[index]["distance_gt_m"]) for index in valid_indices]
        errors = [float(parsed[index]["range_error_m"]) for index in valid_indices]
        group_report: Dict[str, Any] = {
            "session_id": key[0],
            "pose_id": key[1],
            "raw_count": len(indices),
            "base_valid_count": len(valid_indices),
            "base_invalid_count": len(indices) - len(valid_indices),
            "uwb_timing": timestamp_stats,
            "sync_gap_a_s": _compact_stats(sync_a),
            "sync_gap_b_s": _compact_stats(sync_b),
            "range_m": _compact_stats(ranges),
            "distance_gt_m": _compact_stats(distances),
            "range_error_m": _compact_stats(errors),
        }
        if valid_indices:
            translations = [parsed[index]["translation"] for index in valid_indices]
            quaternions = [parsed[index]["quaternion"] for index in valid_indices]
            try:
                position_stability, rotation_stability = _pose_stability(
                    translations, quaternions
                )
                group_report["position_stability_m"] = position_stability
                group_report["rotation_stability_deg"] = rotation_stability
            except (DatasetError, ValueError) as exc:
                group_report["geometry_error"] = str(exc)
            for label, position_fields, quaternion_fields in (
                ("mocap_a", A_POSITION_FIELDS, A_QUATERNION_FIELDS),
                ("mocap_b", B_POSITION_FIELDS, B_QUATERNION_FIELDS),
            ):
                positions = [
                    [_required_finite_float(annotated[index].get(name), name, index)
                     for name in position_fields]
                    for index in valid_indices
                ]
                orientations = [
                    [_required_finite_float(annotated[index].get(name), name, index)
                     for name in quaternion_fields]
                    for index in valid_indices
                ]
                try:
                    position_rms, rotation_rms = _pose_stability(
                        positions, orientations
                    )
                    group_report["{}_position_stability_m".format(label)] = position_rms
                    group_report["{}_rotation_stability_deg".format(label)] = rotation_rms
                except (DatasetError, ValueError) as exc:
                    group_report["{}_stability_error".format(label)] = str(exc)
        if len(valid_indices) < cfg.min_valid_ranges:
            message = "{}/{} has {} valid rows; requires {}".format(
                key[0], key[1], len(valid_indices), cfg.min_valid_ranges
            )
            insufficient.append(message)
        group_reports.append(group_report)

    if duplicate_counts["duplicate_sample_key_count"]:
        severe.append(
            "duplicate measurement sample keys (including attempt_id when present): {}".format(
                duplicate_counts["duplicate_sample_key_count"]
            )
        )
    if duplicate_counts["exact_duplicate_count"]:
        severe.append(
            "exact duplicate rows: {}".format(duplicate_counts["exact_duplicate_count"])
        )
    if not any(item is not None for item in parsed):
        severe.append("no row passed base validity and geometry checks")
    if fail_incomplete_poses and insufficient:
        severe.append(
            "poses below min_valid_ranges: {}".format("; ".join(insufficient))
        )

    session_reports = _session_timing_reports(annotated)
    coverage = _coverage_report([item for item in parsed if item is not None])
    report: Dict[str, Any] = {
        "schema_version": 1,
        "data_provenance": provenance,
        "ingestion_path": ingestion_path,
        "synthetic": (
            _synthetic_tristate(provenance) if provenance != "invalid_mixed" else None
        ),
        "passed": not severe,
        "row_count": len(rows),
        "base_valid_count": sum(item is not None for item in parsed),
        "base_invalid_count": sum(item is None for item in parsed),
        "session_count": len({_text(row.get("session_id")) for row in annotated}),
        "pose_count": len(groups),
        "min_valid_ranges": cfg.min_valid_ranges,
        "configured_id_pair": {
            "source_id": cfg.expected_source_id,
            "peer_id": cfg.expected_peer_id,
            "direction": "source_to_peer",
            "allow_reverse": False,
            "filter_active": cfg.expected_anchor_id is not None,
        },
        "duplicate_counts": duplicate_counts,
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "coverage": coverage,
        "sessions": session_reports,
        "poses": group_reports,
        "severe_errors": severe,
    }
    return RawValidationResult(annotated, report, severe)


def compare_coverage_to_pose_domain(
    coverage: Mapping[str, Any], pose_domain: Mapping[str, Any]
) -> Dict[str, Any]:
    """Compare coordinate-independent observed coverage with configured bounds.

    Relative roll/pitch/yaw are intentionally not reconstructed here because
    the project has not confirmed an Euler sequence.  Their status is made
    explicit instead of silently choosing one.
    """

    axis_mapping = {
        "distance_m": "distance_m",
        "azimuth_deg": "azimuth_deg",
        "elevation_deg": "elevation_deg",
    }
    result: Dict[str, Any] = {"axes": {}, "violations": []}
    for domain_name, coverage_name in axis_mapping.items():
        bounds = pose_domain.get(domain_name)
        if not isinstance(bounds, Mapping) or "min" not in bounds or "max" not in bounds:
            raise DatasetSchemaError(
                "pose domain axis {} must contain min and max".format(domain_name)
            )
        lower = bounds.get("min")
        upper = bounds.get("max")
        observed = coverage.get(coverage_name)
        if lower is None or upper is None:
            result["axes"][domain_name] = {
                "status": "domain_unconfirmed",
                "configured_min": lower,
                "configured_max": upper,
                "observed": observed,
            }
            continue
        lower_value = float(lower)
        upper_value = float(upper)
        if not math.isfinite(lower_value) or not math.isfinite(upper_value):
            raise DatasetSchemaError(
                "pose domain bounds for {} must be finite or null".format(domain_name)
            )
        if lower_value > upper_value:
            raise DatasetSchemaError(
                "pose domain minimum exceeds maximum for {}".format(domain_name)
            )
        if not isinstance(observed, Mapping) or "min" not in observed or "max" not in observed:
            result["axes"][domain_name] = {
                "status": "no_valid_observation",
                "configured_min": lower_value,
                "configured_max": upper_value,
            }
            continue
        observed_min = float(observed["min"])
        observed_max = float(observed["max"])
        inside = observed_min >= lower_value and observed_max <= upper_value
        spans_configured_bounds = observed_min <= lower_value and observed_max >= upper_value
        item = {
            "status": "checked",
            "configured_min": lower_value,
            "configured_max": upper_value,
            "observed_min": observed_min,
            "observed_max": observed_max,
            "observed_within_domain": inside,
            "observed_spans_configured_bounds": spans_configured_bounds,
        }
        result["axes"][domain_name] = item
        if not inside:
            result["violations"].append(
                "observed {} [{}, {}] lies outside configured [{}, {}]".format(
                    domain_name,
                    observed_min,
                    observed_max,
                    lower_value,
                    upper_value,
                )
            )
    for name in ("relative_roll_deg", "relative_pitch_deg", "relative_yaw_deg"):
        bounds = pose_domain.get(name)
        result["axes"][name] = {
            "status": "not_evaluated",
            "configured": dict(bounds) if isinstance(bounds, Mapping) else bounds,
            "reason": "relative Euler sequence is not confirmed; no convention was guessed",
        }
    return result


def aggregate_pose_rows(
    rows: Sequence[Mapping[str, Any]], config: Optional[AggregationConfig] = None
) -> AggregationResult:
    """Aggregate raw measurements into one accepted or rejected row per pose."""

    cfg = (
        AggregationConfig.from_mapping(config)
        if isinstance(config, Mapping)
        else (config or AggregationConfig())
    )
    if not rows:
        raise DatasetSchemaError("cannot aggregate an empty raw dataset")
    provenance = dataset_provenance(rows)
    ingestion_path = canonical_raw_ingestion_path(rows, provenance)
    headers = set().union(*(row.keys() for row in rows))
    missing_keys = sorted(set(POSE_KEY_FIELDS + ("range_m",)) - headers)
    if missing_keys:
        raise DatasetSchemaError(
            "missing aggregation columns: {}".format(", ".join(missing_keys))
        )

    annotated, parsed, duplicate_counts = _annotate_base_rows(rows, cfg)
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for index, row in enumerate(annotated):
        groups[(_text(row.get("session_id")), _text(row.get("pose_id")))].append(index)

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    total_outliers = 0
    for key in sorted(groups):
        indices = groups[key]
        candidates = [index for index in indices if parsed[index] is not None]
        outlier_info = _mark_outliers(annotated, parsed, candidates, cfg.outlier)
        outlier_indices = set(outlier_info["indices"])
        kept = [index for index in candidates if index not in outlier_indices]
        total_outliers += len(outlier_indices)
        pose_row = _aggregate_one_pose(
            key,
            indices,
            candidates,
            kept,
            annotated,
            parsed,
            cfg,
            outlier_info,
        )
        if pose_row["quality_status"] == "accepted":
            accepted.append(pose_row)
        else:
            rejected.append(pose_row)

    summary: Dict[str, Any] = {
        "schema_version": 1,
        "data_provenance": provenance,
        "ingestion_path": ingestion_path,
        "synthetic": _synthetic_tristate(provenance),
        "input_row_count": len(rows),
        "pose_count": len(groups),
        "accepted_pose_count": len(accepted),
        "rejected_pose_count": len(rejected),
        "outlier_count": total_outliers,
        "base_valid_row_count": sum(item is not None for item in parsed),
        "base_rejected_row_count": sum(item is None for item in parsed),
        "duplicate_counts": duplicate_counts,
        "min_valid_ranges": cfg.min_valid_ranges,
        "configured_id_pair": {
            "source_id": cfg.expected_source_id,
            "peer_id": cfg.expected_peer_id,
            "direction": "source_to_peer",
            "allow_reverse": False,
            "filter_active": cfg.expected_source_id is not None,
        },
        "outlier_config": {
            "enabled": cfg.outlier.enabled,
            "method": cfg.outlier.method,
            "field": cfg.outlier.field,
            "threshold": cfg.outlier.threshold,
            "mad_scale": cfg.outlier.mad_scale,
            "min_samples": cfg.outlier.min_samples,
            "minimum_absolute_deviation_m": cfg.outlier.minimum_absolute_deviation_m,
        },
    }
    return AggregationResult(accepted, rejected, annotated, summary)


def fit_normalization(
    train_rows: Sequence[Mapping[str, Any]],
    feature_fields: Sequence[str] = POSE12_FIELDS,
) -> Dict[str, Any]:
    """Fit population mean/std from training rows only."""

    if not train_rows:
        raise DatasetError("cannot fit normalization on an empty training set")
    fields = list(feature_fields)
    columns: List[List[float]] = [[] for _ in fields]
    for row_index, row in enumerate(train_rows):
        for column_index, name in enumerate(fields):
            columns[column_index].append(_required_finite_float(row.get(name), name, row_index))
    means = [statistics.fmean(column) for column in columns]
    minima = [min(column) for column in columns]
    maxima = [max(column) for column in columns]
    raw_std = [math.sqrt(statistics.fmean([(value - mean) ** 2 for value in column]))
               for column, mean in zip(columns, means)]
    zero_variance = [name for name, value in zip(fields, raw_std) if value == 0.0]
    scales = [value if value > 0.0 else 1.0 for value in raw_std]
    return {
        "schema_version": 1,
        "fitted_on": "train",
        "fit_row_count": len(train_rows),
        "feature_order": fields,
        "feature_names": fields,
        "mean": means,
        "std": scales,
        "scale": scales,
        "train_min": minima,
        "train_max": maxima,
        "raw_std": raw_std,
        "mean_by_field": dict(zip(fields, means)),
        "std_by_field": dict(zip(fields, scales)),
        "zero_variance_fields": zero_variance,
    }


def apply_normalization(
    rows: Sequence[Mapping[str, Any]], normalization: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    """Copy rows and append ``norm_<field>`` values using fixed parameters."""

    fields = list(normalization.get("feature_order", []))
    means = list(normalization.get("mean", []))
    scales = list(normalization.get("std", []))
    if not fields or not (len(fields) == len(means) == len(scales)):
        raise DatasetError("normalization has inconsistent feature_order/mean/std")
    output: List[Dict[str, Any]] = []
    for row_index, original in enumerate(rows):
        row = dict(original)
        for name, mean, scale in zip(fields, means, scales):
            scale_value = float(scale)
            if scale_value <= 0.0 or not math.isfinite(scale_value):
                raise DatasetError("normalization scale for {} must be finite and > 0".format(name))
            value = _required_finite_float(row.get(name), name, row_index)
            row["norm_{}".format(name)] = (value - float(mean)) / scale_value
        output.append(row)
    return output


def split_pose_rows(
    rows: Sequence[Mapping[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    split_mode: str = "auto",
    near_translation_m: float = 0.01,
    near_rotation_deg: float = 1.0,
    feature_fields: Sequence[str] = POSE12_FIELDS,
) -> SplitResult:
    """Split aggregate pose rows without session, pose, or near-geometry leakage.

    ``auto`` uses session-preserving connected components when there are at
    least three sessions.  With one or two sessions it explicitly falls back to
    geometry-cluster splitting, which is only a same-environment process check.
    """

    if not rows:
        raise DatasetError("cannot split an empty pose dataset")
    provenance = dataset_provenance(rows)
    ratios = (float(train_ratio), float(val_ratio), float(test_ratio))
    if any(value <= 0.0 or not math.isfinite(value) for value in ratios):
        raise DatasetError("train/val/test ratios must be finite and > 0")
    ratio_sum = sum(ratios)
    ratios = tuple(value / ratio_sum for value in ratios)
    if split_mode not in ("auto", "session", "pose"):
        raise DatasetError("split_mode must be one of: auto, session, pose")
    if near_translation_m < 0.0 or near_rotation_deg < 0.0:
        raise DatasetError("near-duplicate tolerances must be >= 0")

    copied = [dict(row) for row in rows]
    pose_keys: List[Tuple[str, str]] = []
    seen_pose_keys = set()
    sessions: Dict[str, List[int]] = defaultdict(list)
    geometries: List[Tuple[Tuple[float, float, float], List[List[float]]]] = []
    for index, row in enumerate(copied):
        session = _text(row.get("session_id"))
        pose = _text(row.get("pose_id"))
        if not session or not pose:
            raise DatasetSchemaError("every aggregate row needs non-empty session_id and pose_id")
        key = (session, pose)
        if key in seen_pose_keys:
            raise DatasetError("duplicate aggregate pose key: {}/{}".format(*key))
        seen_pose_keys.add(key)
        pose_keys.append(key)
        sessions[session].append(index)
        geometries.append(_row_geometry(row, index))

    geometry_clusters = _near_geometry_clusters(
        geometries, near_translation_m, near_rotation_deg
    )
    warnings: List[str] = []
    mode_used: str
    units: List[List[int]]
    if split_mode == "pose":
        mode_used = "pose_geometry_groups"
        units = geometry_clusters
        warnings.append(
            "pose-level split is a same-environment process check; it is not cross-session evaluation"
        )
    elif split_mode == "session" or len(sessions) >= 3:
        units = _session_component_units(sessions, geometry_clusters)
        if len(units) < 3:
            raise DatasetError(
                "fewer than three independent session components remain after near-duplicate "
                "geometry linking; collect more independent sessions or explicitly request "
                "split_mode='pose' for a process-only evaluation"
            )
        mode_used = "session_components"
    else:
        if split_mode == "session":
            raise DatasetError("session split requires at least three sessions")
        mode_used = "pose_geometry_groups"
        units = geometry_clusters
        warnings.append(
            "only {} session(s): pose/geometry fallback is not evidence of cross-session "
            "or cross-environment generalization".format(len(sessions))
        )

    if len(units) < 3:
        raise DatasetError(
            "at least three independent split units are required for non-empty train/val/test"
        )
    assignments = _assign_units(units, ratios, seed)
    split_indices = {
        name: sorted(index for unit_index in assignments[name] for index in units[unit_index])
        for name in ("train", "val", "test")
    }
    _verify_split_isolation(
        split_indices,
        pose_keys,
        geometries,
        near_translation_m,
        near_rotation_deg,
        mode_used,
        [str(row.get("session_id")) for row in copied],
    )

    raw_train = [copied[index] for index in split_indices["train"]]
    raw_val = [copied[index] for index in split_indices["val"]]
    raw_test = [copied[index] for index in split_indices["test"]]
    normalization = fit_normalization(raw_train, feature_fields)
    train = apply_normalization(raw_train, normalization)
    val = apply_normalization(raw_val, normalization)
    test = apply_normalization(raw_test, normalization)

    def split_description(name: str, selected: Sequence[int]) -> Dict[str, Any]:
        selected_sessions = sorted({pose_keys[index][0] for index in selected})
        return {
            "row_count": len(selected),
            "session_ids": selected_sessions,
            "sessions": selected_sessions,
            "pose_keys": [
                {"session_id": pose_keys[index][0], "pose_id": pose_keys[index][1]}
                for index in selected
            ],
        }

    manifest = {
        "schema_version": 1,
        "data_provenance": provenance,
        "synthetic": _synthetic_tristate(provenance),
        "seed": int(seed),
        "requested_mode": split_mode,
        "split_rule": mode_used,
        "ratios_normalized": {
            "train": ratios[0],
            "val": ratios[1],
            "test": ratios[2],
        },
        "near_duplicate_geometry": {
            "translation_tolerance_m": near_translation_m,
            "rotation_tolerance_deg": near_rotation_deg,
            "cluster_count": len(geometry_clusters),
        },
        "normalization": {
            "fitted_on": "train",
            "feature_order": list(feature_fields),
            "file": "normalization.json",
        },
        "warnings": warnings,
        "train_sessions": sorted(
            {pose_keys[index][0] for index in split_indices["train"]}
        ),
        "splits": {
            "train": split_description("train", split_indices["train"]),
            "val": split_description("val", split_indices["val"]),
            "test": split_description("test", split_indices["test"]),
        },
    }
    return SplitResult(train, val, test, normalization, manifest)


def _annotate_base_rows(
    rows: Sequence[Mapping[str, Any]], cfg: AggregationConfig
) -> Tuple[List[Dict[str, Any]], List[Optional[Dict[str, Any]]], Dict[str, int]]:
    annotated = [dict(row) for row in rows]
    parsed: List[Optional[Dict[str, Any]]] = [None] * len(rows)
    exact_seen: Dict[Tuple[Tuple[str, str], ...], int] = {}
    sample_seen: Dict[Tuple[str, ...], int] = {}
    exact_duplicates = 0
    sample_duplicates = 0
    original_fields = sorted(set().union(*(row.keys() for row in rows)))

    for index, row in enumerate(annotated):
        fingerprint = tuple((name, _text(row.get(name))) for name in original_fields)
        for name in (
            "processing_status",
            "processing_invalid_reason",
            "is_outlier",
            "outlier_reason",
        ):
            if name in row:
                row["source_{}".format(name)] = row.get(name)
        row["processing_status"] = "rejected"
        row["processing_invalid_reason"] = ""
        row["is_outlier"] = "false"
        row["outlier_reason"] = ""
        reasons: List[str] = []

        if fingerprint in exact_seen:
            reasons.append("exact_duplicate_row")
            exact_duplicates += 1
        else:
            exact_seen[fingerprint] = index

        sample_index = _text(row.get("sample_index"))
        sample_key_parts = [
            _text(row.get("session_id")),
            _text(row.get("pose_id")),
        ]
        attempt_id = _text(row.get("attempt_id"))
        if attempt_id:
            sample_key_parts.append(attempt_id)
        sample_key_parts.append(sample_index)
        sample_key = tuple(sample_key_parts)
        if sample_index:
            if sample_key in sample_seen:
                reasons.append("duplicate_sample_key")
                sample_duplicates += 1
            else:
                sample_seen[sample_key] = index

        existing_reason = _text(row.get("invalid_reason"))
        if existing_reason:
            reasons.append("upstream_invalid:" + existing_reason)
        for field_name in POSE_KEY_FIELDS:
            if not _text(row.get(field_name)):
                reasons.append("missing_{}".format(field_name))

        _require_true_flag(row, "is_range_valid", reasons)
        _require_true_flag(row, "is_sync_valid", reasons)
        _require_true_flag(row, "is_stable", reasons)
        if "is_mocap_valid" in row:
            _require_true_flag(row, "is_mocap_valid", reasons)
        if cfg.require_los:
            _check_los(row, reasons)
        _check_id_pair(row, cfg, reasons)

        range_value = _parse_finite_or_reason(row.get("range_m"), "range_m", reasons)
        if range_value is not None:
            if range_value <= cfg.range_min_m:
                reasons.append("range_at_or_below_min")
            if cfg.range_max_m is not None and range_value > cfg.range_max_m:
                reasons.append("range_above_max")

        geometry: Optional[Dict[str, Any]] = None
        try:
            geometry = _geometry_from_row(row, index, cfg)
            if range_value is not None:
                supplied_error = _optional_finite_float(row.get("range_error_m"))
                computed_error = range_value - geometry["distance_gt_m"]
                if supplied_error is not None and abs(supplied_error - computed_error) > max(
                    1.0e-9, cfg.geometry_distance_tolerance_m
                ):
                    reasons.append("range_error_inconsistent")
                geometry["range_error_m"] = computed_error
            supplied_distance = _optional_finite_float(row.get("distance_gt_m"))
            if supplied_distance is not None and abs(
                supplied_distance - geometry["distance_gt_m"]
            ) > cfg.geometry_distance_tolerance_m:
                reasons.append("distance_gt_inconsistent")
            _update_geometry_columns(row, geometry)
        except (DatasetError, ValueError, TypeError) as exc:
            reasons.append("geometry_invalid:" + _single_line(str(exc)))

        reasons = _deduplicate(reasons)
        if reasons:
            row["processing_invalid_reason"] = ";".join(reasons)
            row["processing_status"] = "rejected"
        elif range_value is not None and geometry is not None:
            row["processing_status"] = "candidate"
            parsed[index] = {
                **geometry,
                "range_m": range_value,
                "range_error_m": range_value - geometry["distance_gt_m"],
            }
    return annotated, parsed, {
        "exact_duplicate_count": exact_duplicates,
        "duplicate_sample_key_count": sample_duplicates,
    }


def _geometry_from_row(
    row: Mapping[str, Any], row_index: int, cfg: AggregationConfig
) -> Dict[str, Any]:
    rel_field_names = REL_TRANSLATION_FIELDS + REL_QUATERNION_FIELDS
    rel_presence = [bool(_text(row.get(name))) for name in rel_field_names]
    rel_values = [_optional_finite_float(row.get(name)) for name in REL_TRANSLATION_FIELDS]
    rel_quaternion = [
        _optional_finite_float(row.get(name)) for name in REL_QUATERNION_FIELDS
    ]
    if any(rel_presence) and not all(value is not None for value in rel_values + rel_quaternion):
        raise DatasetSchemaError(
            "row {} has partial or non-finite relative translation/quaternion fields".format(
                row_index
            )
        )
    rel_complete = all(value is not None for value in rel_values + rel_quaternion)
    a_b_fields = A_POSITION_FIELDS + A_QUATERNION_FIELDS + B_POSITION_FIELDS + B_QUATERNION_FIELDS
    a_b_presence = [bool(_text(row.get(name))) for name in a_b_fields]
    a_b_values = [_optional_finite_float(row.get(name)) for name in a_b_fields]
    a_b_complete = all(value is not None for value in a_b_values)
    if any(a_b_presence) and not a_b_complete:
        missing = [name for name, value in zip(a_b_fields, a_b_values) if value is None]
        raise DatasetSchemaError(
            "row {} has partial or non-finite A/B pose fields: {}".format(
                row_index, ", ".join(missing)
            )
        )

    if rel_complete:
        translation = tuple(float(value) for value in rel_values if value is not None)
        quaternion = tuple(float(value) for value in rel_quaternion if value is not None)
    else:
        if not a_b_complete:
            missing = [name for name, value in zip(a_b_fields, a_b_values) if value is None]
            raise DatasetSchemaError(
                "row {} lacks finite relative geometry and A/B pose fields: {}".format(
                    row_index, ", ".join(missing)
                )
            )
        position_a = [float(row[name]) for name in A_POSITION_FIELDS]
        quaternion_a = [float(row[name]) for name in A_QUATERNION_FIELDS]
        position_b = [float(row[name]) for name in B_POSITION_FIELDS]
        quaternion_b = [float(row[name]) for name in B_QUATERNION_FIELDS]
        rel_t, rel_q = relative_pose(position_a, quaternion_a, position_b, quaternion_b)
        translation = tuple(float(value) for value in rel_t)
        quaternion = tuple(float(value) for value in rel_q)

    if len(translation) != 3 or len(quaternion) != 4:
        raise DatasetError("relative pose has unexpected dimensions")
    if not all(math.isfinite(value) for value in translation + quaternion):
        raise DatasetError("relative pose contains a non-finite value")
    quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
    if quaternion_norm <= 1.0e-12:
        raise DatasetError("relative quaternion norm is zero")
    quaternion = tuple(value / quaternion_norm for value in quaternion)
    matrix_raw = quaternion_to_matrix(quaternion)
    matrix = [[float(matrix_raw[row_number][column]) for column in range(3)] for row_number in range(3)]
    _assert_rotation_matrix(
        matrix,
        cfg.rotation_orthogonality_tolerance,
        cfg.rotation_determinant_tolerance,
    )
    if rel_complete and a_b_complete:
        position_a = [float(row[name]) for name in A_POSITION_FIELDS]
        quaternion_a = [float(row[name]) for name in A_QUATERNION_FIELDS]
        position_b = [float(row[name]) for name in B_POSITION_FIELDS]
        quaternion_b = [float(row[name]) for name in B_QUATERNION_FIELDS]
        derived_t, derived_q = relative_pose(
            position_a, quaternion_a, position_b, quaternion_b
        )
        translation_difference = math.sqrt(
            sum(
                (float(derived_t[axis]) - translation[axis]) ** 2
                for axis in range(3)
            )
        )
        derived_matrix_raw = quaternion_to_matrix(derived_q)
        derived_matrix = [
            [float(derived_matrix_raw[r][c]) for c in range(3)] for r in range(3)
        ]
        rotation_difference = _rotation_difference_deg(matrix, derived_matrix)
        if translation_difference > 1.0e-5 or rotation_difference > 1.0e-4:
            raise DatasetError(
                "stored relative pose disagrees with raw A/B poses "
                "(translation {} m, rotation {} deg)".format(
                    _format_number(translation_difference),
                    _format_number(rotation_difference),
                )
            )
    supplied_matrix_values = [
        _optional_finite_float(row.get(name)) for name in REL_MATRIX_FIELDS
    ]
    supplied_matrix_presence = [bool(_text(row.get(name))) for name in REL_MATRIX_FIELDS]
    if any(supplied_matrix_presence):
        if not all(value is not None for value in supplied_matrix_values):
            raise DatasetSchemaError(
                "row {} has partial or non-finite relative rotation matrix fields".format(
                    row_index
                )
            )
        supplied_matrix = [
            [float(supplied_matrix_values[r * 3 + c]) for c in range(3)]
            for r in range(3)
        ]
        _assert_rotation_matrix(
            supplied_matrix,
            cfg.rotation_orthogonality_tolerance,
            cfg.rotation_determinant_tolerance,
        )
        maximum_difference = max(
            abs(supplied_matrix[r][c] - matrix[r][c])
            for r in range(3)
            for c in range(3)
        )
        if maximum_difference > 1.0e-5:
            raise DatasetError(
                "relative quaternion and rotation matrix disagree (max difference {})".format(
                    _format_number(maximum_difference)
                )
            )
    distance = math.sqrt(sum(value * value for value in translation))
    return {
        "translation": translation,
        "quaternion": quaternion,
        "matrix": matrix,
        "distance_gt_m": distance,
    }


def _update_geometry_columns(row: MutableMapping[str, Any], geometry: Mapping[str, Any]) -> None:
    for name, value in zip(REL_TRANSLATION_FIELDS, geometry["translation"]):
        if not _text(row.get(name)):
            row[name] = value
    for name, value in zip(REL_QUATERNION_FIELDS, geometry["quaternion"]):
        if not _text(row.get(name)):
            row[name] = value
    flat_matrix = [value for matrix_row in geometry["matrix"] for value in matrix_row]
    for name, value in zip(REL_MATRIX_FIELDS, flat_matrix):
        if not _text(row.get(name)):
            row[name] = value
    if not _text(row.get("distance_gt_m")):
        row["distance_gt_m"] = geometry["distance_gt_m"]
    if "range_error_m" in geometry and not _text(row.get("range_error_m")):
        row["range_error_m"] = geometry["range_error_m"]


def _mark_outliers(
    annotated: List[Dict[str, Any]],
    parsed: Sequence[Optional[Mapping[str, Any]]],
    candidates: Sequence[int],
    cfg: OutlierConfig,
) -> Dict[str, Any]:
    disabled = not cfg.enabled or cfg.method == "none" or len(candidates) < cfg.min_samples
    values = [float(parsed[index][cfg.field]) for index in candidates if parsed[index] is not None]
    if not values:
        return {"indices": [], "center": None, "mad": None, "cutoff": None}
    center = statistics.median(values)
    mad = statistics.median([abs(value - center) for value in values])
    cutoff = max(
        cfg.threshold * cfg.mad_scale * mad,
        cfg.minimum_absolute_deviation_m,
    )
    outliers: List[int] = []
    if not disabled:
        tolerance = max(1.0e-15, abs(center) * 1.0e-15)
        for index in candidates:
            item = parsed[index]
            assert item is not None
            deviation = abs(float(item[cfg.field]) - center)
            if deviation > cutoff + tolerance:
                annotated[index]["is_outlier"] = "true"
                annotated[index]["outlier_reason"] = (
                    "hampel_mad:{} deviation={} cutoff={}".format(
                        cfg.field, _format_number(deviation), _format_number(cutoff)
                    )
                )
                annotated[index]["processing_status"] = "outlier"
                outliers.append(index)
            else:
                annotated[index]["processing_status"] = "valid"
    else:
        for index in candidates:
            annotated[index]["processing_status"] = "valid"
    return {"indices": outliers, "center": center, "mad": mad, "cutoff": cutoff}


def _aggregate_one_pose(
    key: Tuple[str, str],
    indices: Sequence[int],
    candidates: Sequence[int],
    kept: Sequence[int],
    annotated: List[Dict[str, Any]],
    parsed: Sequence[Optional[Mapping[str, Any]]],
    cfg: AggregationConfig,
    outlier_info: Mapping[str, Any],
) -> Dict[str, Any]:
    environments = sorted(
        {
            _text(annotated[index].get("environment_id"))
            for index in indices
            if _text(annotated[index].get("environment_id"))
        }
    )
    rejection_reasons: List[str] = []
    if len(environments) > 1:
        rejection_reasons.append("inconsistent_environment_id")
    origins = sorted(
        {
            _text(annotated[index].get("data_origin"))
            for index in indices
            if _text(annotated[index].get("data_origin"))
        }
    )
    group_provenance = dataset_provenance([annotated[index] for index in indices])
    group_is_synthetic = _synthetic_tristate(group_provenance)
    if len(origins) > 1:
        rejection_reasons.append("inconsistent_data_origin")
    if len(kept) < cfg.min_valid_ranges:
        rejection_reasons.append(
            "insufficient_valid_ranges:{}<{}".format(len(kept), cfg.min_valid_ranges)
        )

    raw_ranges = [float(parsed[index]["range_m"]) for index in candidates if parsed[index] is not None]
    raw_biases = [
        float(parsed[index]["range_error_m"]) for index in candidates if parsed[index] is not None
    ]
    filtered_ranges = [float(parsed[index]["range_m"]) for index in kept if parsed[index] is not None]
    filtered_biases = [
        float(parsed[index]["range_error_m"]) for index in kept if parsed[index] is not None
    ]
    row: Dict[str, Any] = {name: "" for name in AGGREGATE_COLUMNS}
    row.update(
        {
            "session_id": key[0],
            "pose_id": key[1],
            "environment_id": environments[0] if len(environments) == 1 else "",
            "data_origin": origins[0] if len(origins) == 1 else "",
            "is_synthetic": group_is_synthetic,
            "row_count_total": len(indices),
            "base_rejected_count": len(indices) - len(candidates),
            "range_count_raw": len(candidates),
            "range_count_valid": len(kept),
            "outlier_count": len(candidates) - len(kept),
            "outlier_method": cfg.outlier.method if cfg.outlier.enabled else "none",
            "outlier_center": _none_as_blank(outlier_info.get("center")),
            "outlier_mad": _none_as_blank(outlier_info.get("mad")),
            "outlier_cutoff": _none_as_blank(outlier_info.get("cutoff")),
        }
    )
    _add_distribution_fields(row, "range", raw_ranges, filtered_ranges)
    _add_distribution_fields(row, "bias", raw_biases, filtered_biases)
    if filtered_biases:
        row["bias_sem"] = _sample_std(filtered_biases) / math.sqrt(len(filtered_biases))

    if kept:
        translations = [parsed[index]["translation"] for index in kept if parsed[index] is not None]
        quaternions = [parsed[index]["quaternion"] for index in kept if parsed[index] is not None]
        distances = [float(parsed[index]["distance_gt_m"]) for index in kept if parsed[index] is not None]
        try:
            mean_translation = tuple(
                statistics.fmean(float(value[axis]) for value in translations)
                for axis in range(3)
            )
            mean_quaternion_array = average_quaternions(quaternions)
            mean_quaternion = tuple(float(value) for value in mean_quaternion_array)
            mean_matrix_raw = quaternion_to_matrix(mean_quaternion)
            mean_matrix = [
                [float(mean_matrix_raw[matrix_row][column]) for column in range(3)]
                for matrix_row in range(3)
            ]
            _assert_rotation_matrix(
                mean_matrix,
                cfg.rotation_orthogonality_tolerance,
                cfg.rotation_determinant_tolerance,
            )
            for name, value in zip(REL_TRANSLATION_FIELDS, mean_translation):
                row[name] = value
            for name, value in zip(REL_QUATERNION_FIELDS, mean_quaternion):
                row[name] = value
            for name, value in zip(
                REL_MATRIX_FIELDS, [value for matrix_row in mean_matrix for value in matrix_row]
            ):
                row[name] = value
            row["distance_gt_mean"] = statistics.fmean(distances)
            row["distance_gt_std"] = _sample_std(distances)
            position_stability, rotation_stability = _pose_stability(
                translations, quaternions, mean_translation, mean_matrix
            )
            row["position_stability_m"] = position_stability
            row["rotation_stability_deg"] = rotation_stability
        except (DatasetError, ValueError, TypeError) as exc:
            rejection_reasons.append("rotation_average_invalid:" + _single_line(str(exc)))

        row["mean_fp_rssi"] = _optional_mean(
            annotated[index].get("fp_rssi") for index in kept
        )
        row["mean_rx_rssi"] = _optional_mean(
            annotated[index].get("rx_rssi") for index in kept
        )

    if rejection_reasons:
        row["quality_status"] = "rejected"
        row["rejection_reason"] = ";".join(_deduplicate(rejection_reasons))
        pose_reason = "pose_rejected:" + row["rejection_reason"]
        for index in kept:
            annotated[index]["processing_status"] = "pose_rejected"
            annotated[index]["processing_invalid_reason"] = _join_reason(
                _text(annotated[index].get("processing_invalid_reason")), pose_reason
            )
    else:
        row["quality_status"] = "accepted"
        row["rejection_reason"] = ""
    return row


def _add_distribution_fields(
    row: MutableMapping[str, Any], prefix: str, raw: Sequence[float], filtered: Sequence[float]
) -> None:
    if raw:
        row["{}_mean_raw".format(prefix)] = statistics.fmean(raw)
        row["{}_median_raw".format(prefix)] = statistics.median(raw)
        if prefix == "range":
            # The task-book's unqualified range_median is the before-filter
            # audit statistic; the filtered value has its own explicit field.
            row["range_median"] = statistics.median(raw)
        row["{}_std_raw".format(prefix)] = _sample_std(raw)
        row["{}_mad_raw".format(prefix)] = _median_absolute_deviation(raw)
        for label, probability in (("q05", 0.05), ("q25", 0.25), ("q75", 0.75), ("q95", 0.95)):
            row["{}_{}_raw".format(prefix, label)] = _quantile(raw, probability)
    if filtered:
        row["{}_median_filtered".format(prefix)] = statistics.median(filtered)
        if prefix == "range":
            row["range_mean_filtered"] = statistics.fmean(filtered)
        else:
            row["bias_mean"] = statistics.fmean(filtered)
            row["bias_std"] = _sample_std(filtered)


def _row_geometry(
    row: Mapping[str, Any], row_index: int
) -> Tuple[Tuple[float, float, float], List[List[float]]]:
    translation = tuple(
        _required_finite_float(row.get(name), name, row_index)
        for name in REL_TRANSLATION_FIELDS
    )
    matrix_values = [_optional_finite_float(row.get(name)) for name in REL_MATRIX_FIELDS]
    if all(value is not None for value in matrix_values):
        matrix = [
            [float(matrix_values[row_number * 3 + column]) for column in range(3)]
            for row_number in range(3)
        ]
    else:
        quaternion = [
            _required_finite_float(row.get(name), name, row_index)
            for name in REL_QUATERNION_FIELDS
        ]
        raw_matrix = quaternion_to_matrix(quaternion)
        matrix = [[float(raw_matrix[r][c]) for c in range(3)] for r in range(3)]
    _assert_rotation_matrix(matrix)
    return translation, matrix


def _near_geometry_clusters(
    geometries: Sequence[Tuple[Tuple[float, float, float], List[List[float]]]],
    translation_tolerance: float,
    rotation_tolerance_deg: float,
) -> List[List[int]]:
    parent = list(range(len(geometries)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(len(geometries)):
        for right in range(left + 1, len(geometries)):
            if _geometries_near(
                geometries[left],
                geometries[right],
                translation_tolerance,
                rotation_tolerance_deg,
            ):
                union(left, right)
    clusters: Dict[int, List[int]] = defaultdict(list)
    for index in range(len(geometries)):
        clusters[find(index)].append(index)
    return sorted((sorted(indices) for indices in clusters.values()), key=lambda values: values[0])


def _session_component_units(
    sessions: Mapping[str, Sequence[int]], geometry_clusters: Sequence[Sequence[int]]
) -> List[List[int]]:
    names = sorted(sessions)
    parent = list(range(len(names)))
    by_name = {name: index for index, name in enumerate(names)}
    row_session = {
        row_index: session for session, indices in sessions.items() for row_index in indices
    }

    def find(value: int) -> int:
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for cluster in geometry_clusters:
        cluster_sessions = sorted({row_session[index] for index in cluster})
        for session in cluster_sessions[1:]:
            union(by_name[cluster_sessions[0]], by_name[session])
    component_sessions: Dict[int, List[str]] = defaultdict(list)
    for name in names:
        component_sessions[find(by_name[name])].append(name)
    units = [
        sorted(index for name in component for index in sessions[name])
        for component in component_sessions.values()
    ]
    return sorted(units, key=lambda values: values[0])


def _assign_units(
    units: Sequence[Sequence[int]], ratios: Sequence[float], seed: int
) -> Dict[str, List[int]]:
    names = ("train", "val", "test")
    total_rows = sum(len(unit) for unit in units)
    targets = {name: total_rows * ratio for name, ratio in zip(names, ratios)}
    rng = random.Random(int(seed))
    tie_break = list(range(len(units)))
    rng.shuffle(tie_break)
    rank = {unit_index: order for order, unit_index in enumerate(tie_break)}
    order = sorted(range(len(units)), key=lambda index: (-len(units[index]), rank[index]))
    assignments: Dict[str, List[int]] = {name: [] for name in names}
    counts = {name: 0 for name in names}
    for position, unit_index in enumerate(order):
        remaining = len(order) - position
        empty = [name for name in names if not assignments[name]]
        if remaining == len(empty):
            choices = empty
        else:
            choices = list(names)
        size = len(units[unit_index])

        def score(name: str) -> Tuple[float, int]:
            target = max(targets[name], 1.0e-12)
            before = ((counts[name] - target) / target) ** 2
            after = ((counts[name] + size - target) / target) ** 2
            return after - before, names.index(name)

        selected = min(choices, key=score)
        assignments[selected].append(unit_index)
        counts[selected] += size
    if any(not assignments[name] for name in names):
        raise DatasetError("internal split assignment produced an empty set")
    return assignments


def _verify_split_isolation(
    split_indices: Mapping[str, Sequence[int]],
    pose_keys: Sequence[Tuple[str, str]],
    geometries: Sequence[Tuple[Tuple[float, float, float], List[List[float]]]],
    translation_tolerance: float,
    rotation_tolerance_deg: float,
    mode: str,
    sessions: Sequence[str],
) -> None:
    owner: Dict[int, str] = {}
    for name, indices in split_indices.items():
        for index in indices:
            if index in owner:
                raise DatasetError("row assigned to more than one split")
            owner[index] = name
    if len(owner) != len(pose_keys):
        raise DatasetError("not every pose row was assigned to a split")
    for left in range(len(geometries)):
        for right in range(left + 1, len(geometries)):
            if owner[left] != owner[right] and _geometries_near(
                geometries[left],
                geometries[right],
                translation_tolerance,
                rotation_tolerance_deg,
            ):
                raise DatasetError(
                    "near-duplicate geometry leaked across {} and {}".format(
                        owner[left], owner[right]
                    )
                )
    if mode == "session_components":
        session_owners: Dict[str, str] = {}
        for index, session in enumerate(sessions):
            previous = session_owners.setdefault(session, owner[index])
            if previous != owner[index]:
                raise DatasetError("session {} leaked across splits".format(session))


def _geometries_near(
    left: Tuple[Tuple[float, float, float], List[List[float]]],
    right: Tuple[Tuple[float, float, float], List[List[float]]],
    translation_tolerance: float,
    rotation_tolerance_deg: float,
) -> bool:
    translation_distance = math.sqrt(
        sum((left[0][axis] - right[0][axis]) ** 2 for axis in range(3))
    )
    if translation_distance > translation_tolerance:
        return False
    return _rotation_difference_deg(left[1], right[1]) <= rotation_tolerance_deg


def _pose_stability(
    translations: Sequence[Sequence[float]],
    quaternions: Sequence[Sequence[float]],
    mean_translation: Optional[Sequence[float]] = None,
    mean_matrix: Optional[Sequence[Sequence[float]]] = None,
) -> Tuple[float, float]:
    if not translations or not quaternions or len(translations) != len(quaternions):
        raise DatasetError("pose stability needs matching non-empty translations/quaternions")
    if mean_translation is None:
        mean_translation = [
            statistics.fmean(float(value[axis]) for value in translations) for axis in range(3)
        ]
    if mean_matrix is None:
        mean_quaternion = average_quaternions(quaternions)
        raw = quaternion_to_matrix(mean_quaternion)
        mean_matrix = [[float(raw[r][c]) for c in range(3)] for r in range(3)]
    position_rms = math.sqrt(
        statistics.fmean(
            sum((float(value[axis]) - float(mean_translation[axis])) ** 2 for axis in range(3))
            for value in translations
        )
    )
    angles: List[float] = []
    for quaternion in quaternions:
        raw = quaternion_to_matrix(quaternion)
        matrix = [[float(raw[r][c]) for c in range(3)] for r in range(3)]
        angles.append(_rotation_difference_deg(mean_matrix, matrix))
    rotation_rms = math.sqrt(statistics.fmean(angle * angle for angle in angles))
    return position_rms, rotation_rms


def _assert_rotation_matrix(
    matrix: Sequence[Sequence[float]],
    orthogonality_tolerance: float = 1.0e-6,
    determinant_tolerance: Optional[float] = None,
) -> None:
    if determinant_tolerance is None:
        determinant_tolerance = orthogonality_tolerance
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise DatasetError("rotation matrix must be 3x3")
    values = [float(value) for row in matrix for value in row]
    if not all(math.isfinite(value) for value in values):
        raise DatasetError("rotation matrix contains non-finite values")
    for left in range(3):
        for right in range(3):
            dot = sum(float(matrix[index][left]) * float(matrix[index][right]) for index in range(3))
            expected = 1.0 if left == right else 0.0
            if abs(dot - expected) > orthogonality_tolerance:
                raise DatasetError("rotation matrix is not orthonormal")
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if abs(float(determinant) - 1.0) > determinant_tolerance:
        raise DatasetError("rotation determinant is not +1")


def _rotation_difference_deg(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> float:
    trace = 0.0
    for row in range(3):
        for column in range(3):
            trace += float(left[row][column]) * float(right[row][column])
    cosine = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return math.degrees(math.acos(cosine))


def _add_reason(row: MutableMapping[str, Any], reason: str) -> None:
    row["processing_invalid_reason"] = _join_reason(
        _text(row.get("processing_invalid_reason")), reason
    )


def _check_id_pair(row: Mapping[str, Any], cfg: AggregationConfig, reasons: List[str]) -> None:
    if cfg.expected_source_id is None:
        return
    direct_pair = (_text(row.get("source_id")), _text(row.get("peer_id")))
    alias_pair = (
        _text(row.get("uwb_source_id")),
        _text(row.get("uwb_peer_id")),
    )
    if all(direct_pair):
        actual = direct_pair
    elif all(alias_pair):
        actual = alias_pair
    elif "source_id" in row or "peer_id" in row:
        # Explicit directed columns exist but are incomplete.  Do not reinterpret
        # provisional anchor/node columns as physical roles.
        actual = direct_pair
    elif "uwb_source_id" in row or "uwb_peer_id" in row:
        actual = (
            _text(row.get("uwb_source_id")),
            _text(row.get("uwb_peer_id")),
        )
    else:
        # Compatibility with the task-book's provisional column names.  This
        # fallback is used only when explicit source/peer columns do not exist.
        actual = (_text(row.get("anchor_id")), _text(row.get("node_id")))
    expected = (cfg.expected_source_id, cfg.expected_peer_id)
    if actual != expected:
        reasons.append("id_pair_mismatch")


def _check_los(row: Mapping[str, Any], reasons: List[str]) -> None:
    if "is_los" in row:
        value = _optional_bool(row.get("is_los"))
        if value is not True:
            reasons.append("not_los")
        return
    if "los_status" in row:
        value = _text(row.get("los_status")).lower()
        if value not in ("los", "true", "1", "yes"):
            reasons.append("not_los")
        return
    reasons.append("los_status_missing")


def _require_true_flag(row: Mapping[str, Any], name: str, reasons: List[str]) -> None:
    value = _optional_bool(row.get(name))
    if value is None:
        reasons.append("{}_missing_or_invalid".format(name))
    elif value is not True:
        reasons.append("{}_false".format(name))


def _mapping_bool(values: Mapping[str, Any], name: str, default: bool) -> bool:
    if name not in values:
        return default
    parsed = _optional_bool(values[name])
    if parsed is None:
        raise DatasetError("{} must be a boolean".format(name))
    return parsed


def _optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = _text(value).lower()
    if text in ("true", "1", "yes", "y"):
        return True
    if text in ("false", "0", "no", "n"):
        return False
    return None


def _parse_finite_or_reason(value: Any, name: str, reasons: List[str]) -> Optional[float]:
    parsed = _optional_finite_float(value)
    if parsed is None:
        reasons.append("{}_missing_or_nonfinite".format(name))
    return parsed


def _optional_finite_float(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _required_finite_float(value: Any, name: str, row_index: int) -> float:
    parsed = _optional_finite_float(value)
    if parsed is None:
        raise DatasetSchemaError(
            "row {} field {} must be a finite number".format(row_index, name)
        )
    return parsed


def _sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def _median_absolute_deviation(values: Sequence[float]) -> float:
    if not values:
        raise DatasetError("MAD requires at least one value")
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise DatasetError("quantile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _optional_mean(values: Iterable[Any]) -> Any:
    parsed = _finite_values(values)
    return statistics.fmean(parsed) if parsed else ""


def _finite_values(values: Iterable[Any]) -> List[float]:
    output: List[float] = []
    for value in values:
        parsed = _optional_finite_float(value)
        if parsed is not None:
            output.append(parsed)
    return output


def _compact_stats(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": _sample_std(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _timestamp_statistics(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    differences = [right - left for left, right in zip(values, values[1:])]
    positive = [value for value in differences if value > 0.0]
    mean_interval = statistics.fmean(positive) if positive else None
    return {
        "count": len(values),
        "first_s": values[0],
        "last_s": values[-1],
        "positive_interval_s": _compact_stats(positive),
        "estimated_frequency_hz": (
            1.0 / mean_interval if mean_interval is not None and mean_interval > 0.0 else None
        ),
        "duplicate_timestamp_count": sum(value == 0.0 for value in differences),
        "nonmonotonic_timestamp_count": sum(value < 0.0 for value in differences),
    }


def _session_timing_reports(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_text(row.get("session_id"))].append(row)
    result = []
    for session in sorted(grouped):
        session_rows = grouped[session]
        timestamps = _finite_values(row.get("uwb_timestamp") for row in session_rows)
        receive_timestamps = _finite_values(
            row.get("uwb_receive_timestamp") for row in session_rows
        )
        mocap_a_samples = sorted(
            set(
                _finite_values(
                    value
                    for row in session_rows
                    for value in (
                        row.get("mocap_a_timestamp_before"),
                        row.get("mocap_a_timestamp_after"),
                    )
                )
            )
        )
        mocap_b_samples = sorted(
            set(
                _finite_values(
                    value
                    for row in session_rows
                    for value in (
                        row.get("mocap_b_timestamp_before"),
                        row.get("mocap_b_timestamp_after"),
                    )
                )
            )
        )
        result.append(
            {
                "session_id": session,
                "row_count": len(session_rows),
                "uwb_timing": _timestamp_statistics(timestamps),
                "uwb_receive_timing": _timestamp_statistics(receive_timestamps),
                "mocap_a_observed_sample_timing": _timestamp_statistics(
                    mocap_a_samples
                ),
                "mocap_b_observed_sample_timing": _timestamp_statistics(
                    mocap_b_samples
                ),
            }
        )
    return result


def _coverage_report(parsed: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not parsed:
        return {"valid_geometry_count": 0}
    distances = [float(item["distance_gt_m"]) for item in parsed]
    azimuth = [
        math.degrees(math.atan2(float(item["translation"][1]), float(item["translation"][0])))
        for item in parsed
    ]
    elevation = []
    for item in parsed:
        x, y, z = (float(value) for value in item["translation"])
        elevation.append(math.degrees(math.atan2(z, math.hypot(x, y))))
    return {
        "valid_geometry_count": len(parsed),
        "distance_m": {"min": min(distances), "max": max(distances)},
        "azimuth_deg": {"min": min(azimuth), "max": max(azimuth)},
        "elevation_deg": {"min": min(elevation), "max": max(elevation)},
    }


def dataset_provenance(rows: Sequence[Mapping[str, Any]]) -> str:
    """Return ``synthetic``, ``real``, or ``unknown`` and reject mixed data.

    An absent marker stays unknown; it is never silently promoted to real.
    Explicit synthetic and physical/real rows may not coexist in one artifact.
    """

    synthetic_markers = {
        "synthetic",
        "synthetic_test",
        "test_synthetic",
        "synthetic_unit_test",
    }
    real_markers = {
        "real",
        "experimental",
        "physical_experiment",
        "recorded_real",
        "real_ros_live",
        "real_rosbag_replay",
        "rosbag_replay",
    }
    states = set()
    for row_index, row in enumerate(rows):
        marker = _text(row.get("data_origin")).lower()
        flag = _optional_bool(row.get("is_synthetic"))
        marker_state: Optional[str]
        if marker in synthetic_markers:
            marker_state = "synthetic"
        elif marker in real_markers:
            marker_state = "real"
        else:
            marker_state = None
        flag_state = "synthetic" if flag is True else ("real" if flag is False else None)
        if marker_state is not None and flag_state is not None and marker_state != flag_state:
            raise DatasetError(
                "row {} has conflicting data_origin and is_synthetic provenance".format(
                    row_index
                )
            )
        states.add(marker_state or flag_state or "unknown")
    if len(states) > 1:
        raise DatasetError(
            "mixed dataset provenance is forbidden: {}".format(", ".join(sorted(states)))
        )
    return next(iter(states)) if states else "unknown"


def canonical_raw_ingestion_path(
    rows: Sequence[Mapping[str, Any]], provenance: Optional[str] = None
) -> str:
    """Validate that rows came through the canonical, auditable raw-data path.

    Live ROS CSV output is an operational preview.  Physical experiment data is
    only canonical after conversion from the immutable rosbag.  Synthetic unit
    tests use their own explicit path and must also carry explicit synthetic
    provenance, so an unmarked test artifact cannot be mistaken for real data.
    """

    if not rows:
        raise DatasetSchemaError("cannot validate ingestion_path for empty rows")
    state = provenance if provenance is not None else dataset_provenance(rows)
    paths = {_text(row.get("ingestion_path")).lower() for row in rows}
    if "" in paths:
        raise DatasetSchemaError("every raw row requires a non-empty ingestion_path")
    if len(paths) != 1:
        raise DatasetError(
            "mixed raw ingestion paths are forbidden: {}".format(
                ", ".join(sorted(paths))
            )
        )
    path = next(iter(paths))
    if state == "real":
        if path == "ros_live":
            raise DatasetError(
                "ros_live is an operational preview and cannot build a formal dataset; "
                "convert the immutable rosbag and use ingestion_path=rosbag_replay"
            )
        if path != "rosbag_replay":
            raise DatasetError(
                "physical raw data requires ingestion_path=rosbag_replay, got {!r}".format(
                    path
                )
            )
    elif state == "synthetic":
        if path != "synthetic_unit_test":
            raise DatasetError(
                "synthetic raw data requires ingestion_path=synthetic_unit_test, got {!r}".format(
                    path
                )
            )
    else:
        raise DatasetError(
            "raw data provenance must be explicitly real or synthetic before dataset construction"
        )
    return path


def _synthetic_tristate(provenance: str) -> Optional[bool]:
    if provenance == "synthetic":
        return True
    if provenance == "real":
        return False
    if provenance == "unknown":
        return None
    raise DatasetError("invalid provenance state: {}".format(provenance))


def _split_reasons(value: str) -> List[str]:
    return [item for item in value.split(";") if item]


def _deduplicate(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))


def _join_reason(existing: str, new_reason: str) -> str:
    return ";".join(_deduplicate(_split_reasons(existing) + [new_reason]))


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _single_line(value: str) -> str:
    return " ".join(value.splitlines()).strip()


def _format_number(value: float) -> str:
    return format(value, ".12g")


def _none_as_blank(value: Any) -> Any:
    return "" if value is None else value


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".17g")
    return value


def _resolved_path(path: os.PathLike[str] | str) -> Path:
    return Path(path).expanduser().resolve(strict=False)


__all__ = [
    "AGGREGATE_COLUMNS",
    "ANNOTATION_COLUMNS",
    "AggregationConfig",
    "AggregationResult",
    "DatasetError",
    "DatasetSchemaError",
    "OutlierConfig",
    "OutputExistsError",
    "POSE12_FIELDS",
    "RAW_REQUIRED_COLUMNS",
    "RawValidationResult",
    "SplitResult",
    "aggregate_pose_rows",
    "canonical_raw_ingestion_path",
    "apply_normalization",
    "assert_distinct_new_outputs",
    "compare_coverage_to_pose_domain",
    "dataset_provenance",
    "file_sha256",
    "fit_normalization",
    "load_config_mapping",
    "ordered_fieldnames",
    "read_csv_rows",
    "select_raw_test_rows",
    "split_pose_rows",
    "validate_raw_rows",
    "write_csv_exclusive",
    "write_json_exclusive",
]
