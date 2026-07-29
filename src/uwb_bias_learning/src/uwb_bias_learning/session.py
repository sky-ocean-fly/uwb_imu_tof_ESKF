"""Create immutable session layouts and append auditable lifecycle events."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .config import (
    ConfigError,
    copy_exclusive,
    create_new_directory,
    load_collection_config,
    load_yaml,
    open_exclusive,
    resolve_config_path,
    validate_ros_topics_config,
    without_private_metadata,
    write_json_exclusive,
)


SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
REQUIRED_METADATA = (
    "data_origin",
    "is_synthetic",
    "environment_id",
    "location",
    "operator",
    "los",
    "uwb.model",
    "uwb.firmware",
    "uwb.ranging_mode",
    "mocap.rigid_body_a",
    "mocap.rigid_body_b",
    "mocap.antenna_mount_description",
)

ALLOWED_DATA_ORIGINS = {
    "physical_experiment": False,
    "synthetic_test": True,
}


def _nested(value: Mapping[str, Any], dotted: str):
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def validate_session_metadata(metadata: Mapping[str, Any]) -> None:
    missing = [
        key
        for key in REQUIRED_METADATA
        if _nested(metadata, key) is None or _nested(metadata, key) == ""
    ]
    if missing:
        raise ConfigError("session metadata is incomplete: %s" % ", ".join(missing))
    if metadata.get("los") is not True:
        raise ConfigError("this project records LOS sessions only; metadata.los must be true")
    data_origin = metadata.get("data_origin")
    if data_origin not in ALLOWED_DATA_ORIGINS:
        raise ConfigError(
            "metadata.data_origin must be one of: %s"
            % ", ".join(sorted(ALLOWED_DATA_ORIGINS))
        )
    is_synthetic = metadata.get("is_synthetic")
    if not isinstance(is_synthetic, bool):
        raise ConfigError("metadata.is_synthetic must be an explicit YAML boolean")
    if is_synthetic is not ALLOWED_DATA_ORIGINS[data_origin]:
        raise ConfigError(
            "metadata.data_origin=%s requires is_synthetic=%s"
            % (data_origin, str(ALLOWED_DATA_ORIGINS[data_origin]).lower())
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest_metadata(session_dir: os.PathLike, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Bind mutable-looking manifest metadata to its session input snapshot."""

    directory = Path(session_dir).expanduser().resolve()
    snapshot = directory / "config_snapshot" / "session_metadata_input.yaml"
    if not snapshot.is_file():
        raise ConfigError("session metadata snapshot is missing: %s" % snapshot)
    loaded = load_yaml(snapshot)
    validate_session_metadata(loaded)
    expected = without_private_metadata(loaded)
    actual = manifest.get("metadata")
    if not isinstance(actual, Mapping) or dict(actual) != expected:
        raise ConfigError("session manifest metadata differs from its immutable input snapshot")
    expected_hash = manifest.get("metadata_snapshot_sha256")
    actual_hash = _sha256(snapshot)
    if expected_hash != actual_hash:
        raise ConfigError("session metadata snapshot SHA-256 does not match the manifest")
    return dict(expected)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def git_commit(workspace: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(workspace),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def append_event(session_dir: os.PathLike, event: str, **details: Any) -> Path:
    path = Path(session_dir).resolve() / "session_events.jsonl"
    payload = {"timestamp_utc": utc_now(), "event": str(event), "details": details}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path


def prepare_session(
    collection_config_path: os.PathLike,
    metadata_path: os.PathLike,
    session_id: str,
    output_root: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    if not SESSION_ID_RE.match(session_id):
        raise ConfigError(
            "session_id must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_' or '-'"
        )
    collection = load_collection_config(collection_config_path, live=True)
    metadata = load_yaml(metadata_path)
    validate_session_metadata(metadata)
    configured_root = collection["output"]["raw_root"]
    root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else resolve_config_path(collection, configured_root)
    )
    root.mkdir(parents=True, exist_ok=True)
    session_dir = create_new_directory(root / session_id)
    snapshots = session_dir / "config_snapshot"
    snapshots.mkdir()
    copy_exclusive(collection["_config_path"], snapshots / "collection.yaml")
    copy_exclusive(collection["_ros_topics_path"], snapshots / "ros_topics.yaml")
    copy_exclusive(metadata["_config_path"], snapshots / "session_metadata_input.yaml")

    workspace = Path(__file__).resolve().parents[4]
    resolved = {
        "schema_version": 1,
        "session_id": session_id,
        "created_utc": utc_now(),
        "workspace": str(workspace),
        "git_commit": git_commit(workspace),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "python_version": platform.python_version(),
        "topics": {
            role: collection["ros_topics"][role]["topic"]
            for role in ("uwb", "mocap_a", "mocap_b", "status")
        },
        "uwb_pair": {
            "source_id": collection["ros_topics"]["uwb"]["source_id"],
            "peer_id": collection["ros_topics"]["uwb"]["peer_id"],
            "direction": collection["ros_topics"]["uwb"]["direction"],
        },
        "metadata": without_private_metadata(metadata),
        "metadata_snapshot_sha256": _sha256(snapshots / "session_metadata_input.yaml"),
    }
    write_json_exclusive(session_dir / "session_manifest.json", resolved)
    append_event(session_dir, "SESSION_CREATED")
    return {
        "session_dir": str(session_dir),
        "bag_prefix": str(session_dir / "raw_topics"),
        "topics": resolved["topics"],
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--collection-config", required=True)
    prepare.add_argument("--metadata", required=True)
    prepare.add_argument("--session-id", required=True)
    prepare.add_argument("--output-root")
    event = sub.add_parser("event")
    event.add_argument("--session-dir", required=True)
    event.add_argument("--name", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_session(
                args.collection_config,
                args.metadata,
                args.session_id,
                args.output_root,
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        else:
            append_event(args.session_dir, args.name)
        return 0
    except (ConfigError, FileExistsError, OSError) as exc:
        print("session operation failed: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
