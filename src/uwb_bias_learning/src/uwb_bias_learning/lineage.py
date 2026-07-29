"""Content-bound split lineage and real/synthetic provenance checks."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Set, Tuple


class LineageError(ValueError):
    """Raised when artifacts do not belong to one fixed dataset split."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load_mapping(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise LineageError("JSON root must be an object: {}".format(path))
    return payload


def _valid_digest(value: Any) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise LineageError("manifest contains an invalid SHA-256 digest")
    return digest


def csv_pose_keys(path: Path) -> Set[Tuple[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise LineageError("CSV has no header: {}".format(path))
        missing = [name for name in ("session_id", "pose_id") if name not in reader.fieldnames]
        if missing:
            raise LineageError(
                "{} is missing lineage keys: {}".format(path, ", ".join(missing))
            )
        keys = []
        for row_number, row in enumerate(reader, start=2):
            key = (str(row["session_id"]).strip(), str(row["pose_id"]).strip())
            if not all(key):
                raise LineageError("empty session_id/pose_id at {} row {}".format(path, row_number))
            keys.append(key)
    if not keys:
        raise LineageError("CSV has no rows: {}".format(path))
    if len(keys) != len(set(keys)):
        raise LineageError("CSV contains duplicate session_id + pose_id rows: {}".format(path))
    return set(keys)


@dataclass(frozen=True)
class SplitContract:
    path: Path
    payload: Mapping[str, Any]
    sha256: str
    data_provenance: str

    @classmethod
    def load(cls, path: Path, allow_synthetic: bool = False) -> "SplitContract":
        resolved = Path(path).expanduser().resolve()
        payload = _load_mapping(resolved)
        provenance = str(payload.get("data_provenance", "")).strip().lower()
        if provenance not in ("real", "synthetic"):
            raise LineageError(
                "split manifest data_provenance must be explicitly real or synthetic; got {!r}".format(
                    provenance or None
                )
            )
        if provenance == "synthetic" and not allow_synthetic:
            raise LineageError(
                "synthetic split is test-only; pass --allow-synthetic explicitly for a software test run"
            )
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise LineageError("split manifest has no artifacts hash mapping")
        return cls(resolved, payload, file_sha256(resolved), provenance)

    def artifact(self, name: str) -> Mapping[str, Any]:
        artifacts = self.payload.get("artifacts", {})
        entry = artifacts.get(name) if isinstance(artifacts, Mapping) else None
        if not isinstance(entry, Mapping):
            raise LineageError("split manifest has no artifact entry {!r}".format(name))
        _valid_digest(entry.get("sha256"))
        return entry

    def artifact_path(self, name: str) -> Path:
        entry = self.artifact(name)
        raw = entry.get("file")
        if raw in (None, ""):
            raise LineageError("artifact {!r} has no file path".format(name))
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = self.path.parent / path
        return path.resolve()

    def verify_artifact(self, name: str, path: Path) -> str:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise LineageError("artifact does not exist: {}".format(source))
        expected = _valid_digest(self.artifact(name).get("sha256"))
        actual = file_sha256(source)
        if actual != expected:
            raise LineageError(
                "{} SHA-256 does not match split manifest artifact {!r}".format(
                    source, name
                )
            )
        return actual

    def pose_keys(self, split: str) -> Set[Tuple[str, str]]:
        splits = self.payload.get("splits")
        node = splits.get(split) if isinstance(splits, Mapping) else None
        if not isinstance(node, Mapping) or not isinstance(node.get("pose_keys"), list):
            raise LineageError("split manifest lacks splits.{}.pose_keys".format(split))
        keys = []
        for item in node["pose_keys"]:
            if not isinstance(item, Mapping):
                raise LineageError("invalid pose key in split {!r}".format(split))
            key = (str(item.get("session_id", "")).strip(), str(item.get("pose_id", "")).strip())
            if not all(key):
                raise LineageError("empty pose key in split {!r}".format(split))
            keys.append(key)
        if not keys or len(keys) != len(set(keys)):
            raise LineageError("split {!r} has empty or duplicate pose keys".format(split))
        return set(keys)

    def verify_pose_csv(self, split: str, path: Path) -> str:
        digest = self.verify_artifact(split, path)
        actual_keys = csv_pose_keys(path)
        if actual_keys != self.pose_keys(split):
            raise LineageError(
                "{} pose keys do not exactly match manifest split {!r}".format(path, split)
            )
        return digest

    def base_lineage(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "split_manifest_sha256": self.sha256,
            "data_provenance": self.data_provenance,
            "train_sha256": _valid_digest(self.artifact("train").get("sha256")),
            "val_sha256": _valid_digest(self.artifact("val").get("sha256")),
            "test_sha256": _valid_digest(self.artifact("test").get("sha256")),
            "split_normalization_sha256": _valid_digest(
                self.artifact("normalization").get("sha256")
            ),
        }


def require_lineage_match(
    recorded: Mapping[str, Any],
    expected: Mapping[str, Any],
    fields: Sequence[str],
    label: str,
) -> None:
    if not isinstance(recorded, Mapping):
        raise LineageError("{} has no lineage mapping".format(label))
    for field in fields:
        if recorded.get(field) != expected.get(field):
            raise LineageError(
                "{} lineage field {!r} does not match the fixed split/run".format(
                    label, field
                )
            )


__all__ = [
    "LineageError",
    "SplitContract",
    "csv_pose_keys",
    "file_sha256",
    "require_lineage_match",
]
