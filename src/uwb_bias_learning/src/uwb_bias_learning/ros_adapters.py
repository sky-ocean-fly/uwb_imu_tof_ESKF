"""Small, auditable adapters around confirmed ROS message schemas.

Importing this module does not require a running ROS master.  ROS-specific class
loading is deferred until :func:`load_ros_message_class` is called so offline
dataset and model tools remain usable without hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class RosAdapterError(ValueError):
    """Raised when a message does not match the configured, confirmed schema."""


@dataclass(frozen=True)
class PoseObservation:
    timestamp: float
    receive_timestamp: float
    frame_id: str
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    sequence: Optional[int]
    timestamp_source: str = "header_stamp"


@dataclass(frozen=True)
class UwbObservation:
    receive_timestamp: float
    source_id: int
    peer_id: int
    range_m: float
    source_role: Optional[int]
    peer_role: Optional[int]
    local_time_raw: Optional[int]
    system_time_raw: Optional[int]
    fp_rssi: Optional[float]
    rx_rssi: Optional[float]
    timestamp: float
    timestamp_source: str


@dataclass(frozen=True)
class RangeExtraction:
    observation: Optional[UwbObservation]
    invalid_reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.observation is not None and not self.invalid_reason


def ros_time_to_seconds(value: Any) -> float:
    """Convert rospy.Time-like or secs/nsecs-like values to seconds."""

    if value is None:
        raise RosAdapterError("timestamp is missing")
    if hasattr(value, "to_sec"):
        return float(value.to_sec())
    if hasattr(value, "secs") and hasattr(value, "nsecs"):
        return float(value.secs) + float(value.nsecs) * 1.0e-9
    if isinstance(value, (int, float, np.number)):
        return float(value)
    raise RosAdapterError("unsupported timestamp value: %r" % (value,))


def extract_pose_stamped(
    message: Any,
    receive_timestamp: float,
    require_nonzero_stamp: bool = True,
) -> PoseObservation:
    """Extract a ``geometry_msgs/PoseStamped`` without changing conventions."""

    try:
        header = message.header
        pose = message.pose
        stamp = ros_time_to_seconds(header.stamp)
        position = np.asarray(
            [pose.position.x, pose.position.y, pose.position.z], dtype=np.float64
        )
        quaternion = np.asarray(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=np.float64,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise RosAdapterError("message is not a usable PoseStamped: %s" % exc)

    if require_nonzero_stamp and stamp <= 0.0:
        raise RosAdapterError("PoseStamped header.stamp is zero")
    if not np.all(np.isfinite(position)):
        raise RosAdapterError("PoseStamped position contains a non-finite value")
    if not np.all(np.isfinite(quaternion)):
        raise RosAdapterError("PoseStamped quaternion contains a non-finite value")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise RosAdapterError("PoseStamped quaternion has zero norm")
    quaternion = quaternion / norm
    sequence = getattr(header, "seq", None)
    return PoseObservation(
        timestamp=stamp,
        receive_timestamp=float(receive_timestamp),
        frame_id=str(getattr(header, "frame_id", "")),
        position=position,
        quaternion_xyzw=quaternion,
        sequence=int(sequence) if sequence is not None else None,
    )


def _optional_number(value: Any, cast):
    if value is None:
        return None
    return cast(value)


def extract_nodeframe2_pair(
    message: Any,
    source_id: int,
    peer_id: int,
    receive_timestamp: float,
    timestamp_source: str = "receive_time",
    mapped_device_timestamp: Optional[float] = None,
) -> RangeExtraction:
    """Extract one directed range from a confirmed Linktrack NodeFrame2 schema.

    ``source_id`` always refers to the outer ``message.id`` and ``peer_id`` to
    one nested ``message.nodes[i].id``.  Reversing this direction is a config
    change, never an implicit fallback.
    """

    try:
        observed_source = int(message.id)
        nodes = list(message.nodes)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RosAdapterError("message is not LinktrackNodeframe2-compatible: %s" % exc)

    if observed_source != int(source_id):
        return RangeExtraction(None, "not_target_source")

    matches = []
    for node in nodes:
        try:
            if int(node.id) == int(peer_id):
                matches.append(node)
        except (AttributeError, TypeError, ValueError):
            continue
    if not matches:
        return RangeExtraction(None, "target_peer_missing")
    if len(matches) != 1:
        return RangeExtraction(None, "duplicate_target_peer")

    node = matches[0]
    try:
        range_m = float(node.dis)
    except (AttributeError, TypeError, ValueError) as exc:
        return RangeExtraction(None, "invalid_range_field:%s" % exc)

    if timestamp_source == "receive_time":
        timestamp = float(receive_timestamp)
    elif timestamp_source == "mapped_device_time":
        if mapped_device_timestamp is None:
            return RangeExtraction(None, "mapped_device_timestamp_missing")
        timestamp = float(mapped_device_timestamp)
    else:
        raise RosAdapterError("unsupported UWB timestamp source: %s" % timestamp_source)

    observation = UwbObservation(
        receive_timestamp=float(receive_timestamp),
        source_id=observed_source,
        peer_id=int(peer_id),
        range_m=range_m,
        source_role=_optional_number(getattr(message, "role", None), int),
        peer_role=_optional_number(getattr(node, "role", None), int),
        local_time_raw=_optional_number(getattr(message, "local_time", None), int),
        system_time_raw=_optional_number(getattr(message, "system_time", None), int),
        fp_rssi=_optional_number(getattr(node, "fp_rssi", None), float),
        rx_rssi=_optional_number(getattr(node, "rx_rssi", None), float),
        timestamp=timestamp,
        timestamp_source=timestamp_source,
    )
    return RangeExtraction(observation)


SUPPORTED_UWB_ADAPTERS = {
    "nodeframe2": extract_nodeframe2_pair,
}


def get_uwb_adapter(name: str):
    try:
        return SUPPORTED_UWB_ADAPTERS[name]
    except KeyError:
        raise RosAdapterError(
            "unsupported or unconfirmed UWB adapter %r; supported after schema "
            "confirmation: %s" % (name, ", ".join(sorted(SUPPORTED_UWB_ADAPTERS)))
        )


def load_ros_message_class(type_name: str):
    """Load a generated ROS message class by ``package/Message`` name."""

    try:
        from roslib.message import get_message_class
    except ImportError as exc:
        raise RosAdapterError("ROS Python environment is not sourced: %s" % exc)
    cls = get_message_class(type_name)
    if cls is None:
        raise RosAdapterError(
            "cannot load ROS message class %s; build and source the workspace" % type_name
        )
    return cls


def message_type_name(message_or_class: Any) -> Optional[str]:
    return getattr(message_or_class, "_type", None)
