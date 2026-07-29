"""Configuration and non-overwriting filesystem helpers.

All paths are resolved relative to the YAML file that declares them.  Live ROS
configuration is intentionally strict: unresolved Gate 1 values must remain
``null`` and cause a clear error rather than silently selecting an old topic or
node ID.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import yaml


class ConfigError(ValueError):
    """Raised when a configuration is missing or internally inconsistent."""


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PACKAGE_ROOT / "configs"


def load_yaml(path: os.PathLike) -> Dict[str, Any]:
    """Load a YAML mapping without applying implicit project defaults."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError("configuration file does not exist: %s" % config_path)
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError("top-level YAML value must be a mapping: %s" % config_path)
    data = dict(data)
    data["_config_path"] = str(config_path)
    return data


def config_directory(config: Mapping[str, Any]) -> Path:
    raw = config.get("_config_path")
    if not raw:
        raise ConfigError("configuration has no _config_path metadata")
    return Path(str(raw)).resolve().parent


def resolve_config_path(config: Mapping[str, Any], value: os.PathLike) -> Path:
    """Resolve ``value`` relative to the declaring configuration file."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config_directory(config) / path).resolve()


def get_nested(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def require_values(config: Mapping[str, Any], dotted_keys: Iterable[str]) -> None:
    missing = []
    for key in dotted_keys:
        value = get_nested(config, key)
        if value is None or value == "" or value == []:
            missing.append(key)
    if missing:
        raise ConfigError(
            "unresolved required configuration values: %s. "
            "Keep them null until confirmed, then fill them before a live run."
            % ", ".join(sorted(missing))
        )


LIVE_ROS_REQUIRED_FIELDS = (
    "uwb.topic",
    "uwb.message_type",
    "uwb.adapter",
    "uwb.source_id",
    "uwb.peer_id",
    "uwb.direction",
    "uwb.timestamp_source",
    "mocap_a.topic",
    "mocap_b.topic",
    "mocap_a.message_type",
    "mocap_b.message_type",
    "status.topic",
    "status.message_type",
    "extrinsics.rigid_body_origins_at_antenna_centers",
    "extrinsics.rigid_body_axes_aligned_with_antennas",
)


def validate_ros_topics_config(config: Mapping[str, Any], live: bool = False) -> None:
    """Validate schema and, optionally, all values needed for live collection."""

    for section in (
        "uwb",
        "mocap_a",
        "mocap_b",
        "status",
        "time",
        "extrinsics",
    ):
        if not isinstance(config.get(section), Mapping):
            raise ConfigError("missing mapping section: %s" % section)
    if live:
        require_values(config, LIVE_ROS_REQUIRED_FIELDS)
        if config["uwb"]["direction"] != "source_to_peer":
            raise ConfigError("uwb.direction must be exactly 'source_to_peer'")
        if config["uwb"]["adapter"] != "nodeframe2":
            raise ConfigError(
                "only the source-confirmed 'nodeframe2' UWB adapter is implemented"
            )
        if config["uwb"]["message_type"] != "nlink_parser/LinktrackNodeframe2":
            raise ConfigError(
                "nodeframe2 adapter requires nlink_parser/LinktrackNodeframe2"
            )
        for role in ("mocap_a", "mocap_b"):
            if config[role]["message_type"] != "geometry_msgs/PoseStamped":
                raise ConfigError(
                    "%s.message_type must be geometry_msgs/PoseStamped" % role
                )
        if config["uwb"]["timestamp_source"] not in (
            "receive_time",
            "mapped_device_time",
        ):
            raise ConfigError(
                "uwb.timestamp_source must be receive_time or mapped_device_time"
            )
        if config["uwb"]["source_id"] == config["uwb"]["peer_id"]:
            raise ConfigError("uwb.source_id and uwb.peer_id must differ")
        if config["status"]["message_type"] != "std_msgs/String":
            raise ConfigError("status.message_type must be exactly 'std_msgs/String'")
        topics = [config[role]["topic"] for role in ("uwb", "mocap_a", "mocap_b", "status")]
        if len(set(topics)) != len(topics):
            raise ConfigError("uwb, mocap and status topics must be distinct")
        if config["time"].get("require_use_sim_time_false") is not True:
            raise ConfigError("time.require_use_sim_time_false must be true")
        if config["time"].get("require_nonzero_mocap_header_stamp") is not True:
            raise ConfigError("time.require_nonzero_mocap_header_stamp must be true")
        if config["time"].get("require_same_mocap_frame") is not True:
            raise ConfigError("time.require_same_mocap_frame must be true")
        if config["extrinsics"]["rigid_body_origins_at_antenna_centers"] is not True:
            raise ConfigError(
                "unit translation extrinsics were not explicitly confirmed true"
            )
        if config["extrinsics"]["rigid_body_axes_aligned_with_antennas"] is not True:
            raise ConfigError("unit rotation extrinsics were not explicitly confirmed true")


def load_collection_config(path: os.PathLike, live: bool = False) -> Dict[str, Any]:
    collection = load_yaml(path)
    require_values(
        collection,
        (
            "ros_topics_config",
            "target_valid_ranges",
            "min_valid_ranges",
            "settling_time_s",
            "stability_window_s",
            "max_position_std_m",
            "max_rotation_std_deg",
            "max_sync_gap_s",
            "output.raw_root",
        ),
    )
    topics_path = resolve_config_path(collection, collection["ros_topics_config"])
    topics = load_yaml(topics_path)
    validate_ros_topics_config(topics, live=live)
    result = dict(collection)
    result["ros_topics"] = topics
    result["_ros_topics_path"] = str(topics_path)
    return result


def load_pose_domain(path: os.PathLike, require_complete: bool = False) -> Dict[str, Any]:
    domain = load_yaml(path)
    axes = (
        "distance_m",
        "azimuth_deg",
        "elevation_deg",
        "relative_roll_deg",
        "relative_pitch_deg",
        "relative_yaw_deg",
    )
    for axis in axes:
        value = domain.get(axis)
        if not isinstance(value, Mapping) or "min" not in value or "max" not in value:
            raise ConfigError("pose domain axis must contain min/max: %s" % axis)
        if require_complete and (value["min"] is None or value["max"] is None):
            raise ConfigError("pose domain is not confirmed: %s" % axis)
        if value["min"] is not None and value["max"] is not None:
            if float(value["min"]) > float(value["max"]):
                raise ConfigError("pose domain min exceeds max: %s" % axis)
    return domain


def without_private_metadata(value: Any) -> Any:
    """Return a JSON/YAML-safe copy without internal ``_`` keys."""

    if isinstance(value, Mapping):
        return {
            str(key): without_private_metadata(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [without_private_metadata(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def create_new_directory(path: os.PathLike) -> Path:
    """Create a directory and fail if it already exists."""

    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=False)
    return target


def open_exclusive(path: os.PathLike, mode: str = "x", **kwargs: Any):
    """Open a new file, rejecting any mode that could overwrite existing data."""

    if "x" not in mode:
        raise ValueError("open_exclusive requires an exclusive creation mode")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.open(mode, **kwargs)


def write_json_exclusive(path: os.PathLike, payload: Any) -> Path:
    target = Path(path).expanduser().resolve()
    with open_exclusive(target, "x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return target


def write_yaml_exclusive(path: os.PathLike, payload: Any) -> Path:
    target = Path(path).expanduser().resolve()
    with open_exclusive(target, "x", encoding="utf-8") as stream:
        yaml.safe_dump(
            without_private_metadata(payload),
            stream,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
    return target


def copy_exclusive(source: os.PathLike, destination: os.PathLike) -> Path:
    src = Path(source).expanduser().resolve()
    dst = Path(destination).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as reader, dst.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
    return dst


def snapshot_configs(config_paths: Sequence[os.PathLike], directory: os.PathLike) -> None:
    target_dir = Path(directory).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=False)
    used_names = set()
    for raw_path in config_paths:
        src = Path(raw_path).expanduser().resolve()
        name = src.name
        if name in used_names:
            raise ConfigError("duplicate snapshot filename: %s" % name)
        used_names.add(name)
        copy_exclusive(src, target_dir / name)
