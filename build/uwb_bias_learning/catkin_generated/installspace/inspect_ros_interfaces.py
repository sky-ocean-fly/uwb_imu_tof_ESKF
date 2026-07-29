#!/usr/bin/env python3
"""Inspect configured ROS interfaces without changing ROS state."""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections import Counter
from pathlib import Path

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from uwb_bias_learning.config import (  # noqa: E402
    ConfigError,
    load_yaml,
    validate_ros_topics_config,
    write_json_exclusive,
)
from uwb_bias_learning.ros_adapters import (  # noqa: E402
    RosAdapterError,
    extract_pose_stamped,
    load_ros_message_class,
)


def _rate(timestamps):
    if len(timestamps) < 2:
        return None
    span = timestamps[-1] - timestamps[0]
    return (len(timestamps) - 1) / span if span > 0.0 else None


def _interval_summary(timestamps):
    if len(timestamps) < 2:
        return {"count": 0}
    intervals = sorted(b - a for a, b in zip(timestamps[:-1], timestamps[1:]))

    def percentile(fraction):
        index = int(round(fraction * (len(intervals) - 1)))
        return intervals[max(0, min(index, len(intervals) - 1))]

    return {
        "count": len(intervals),
        "min_s": intervals[0],
        "median_s": percentile(0.5),
        "p95_s": percentile(0.95),
        "max_s": intervals[-1],
    }


class TopicProbe:
    def __init__(self, role, configured_type):
        self.role = role
        self.configured_type = configured_type
        self.lock = threading.Lock()
        self.receive_times = []
        self.header_times = []
        self.frame_ids = Counter()
        self.sequence_missing = 0
        self._last_sequence = None
        self.zero_header_stamp = 0
        self.invalid_pose = Counter()
        self.quaternion_norms = []
        self.header_delays = []
        self.source_ids = Counter()
        self.peer_ids = Counter()
        self.empty_node_arrays = 0
        self.range_count = 0
        self.nonfinite_ranges = 0
        self.nonpositive_ranges = 0
        self.has_fp_rssi = False
        self.has_rx_rssi = False
        self.has_cir_quality = False
        self.local_time_first = None
        self.local_time_last = None
        self.system_time_first = None
        self.system_time_last = None

    def callback(self, message):
        import rospy

        receive = float(rospy.Time.now().to_sec())
        with self.lock:
            self.receive_times.append(receive)
            if self.role.startswith("mocap"):
                try:
                    observation = extract_pose_stamped(message, receive)
                except RosAdapterError as exc:
                    self.invalid_pose[str(exc)] += 1
                    return
                self.header_times.append(observation.timestamp)
                self.frame_ids[observation.frame_id] += 1
                self.quaternion_norms.append(
                    float(sum(v * v for v in observation.quaternion_xyzw) ** 0.5)
                )
                self.header_delays.append(receive - observation.timestamp)
                if observation.timestamp <= 0.0:
                    self.zero_header_stamp += 1
                sequence = observation.sequence
                if sequence is not None and self._last_sequence is not None:
                    if sequence > self._last_sequence + 1:
                        self.sequence_missing += sequence - self._last_sequence - 1
                self._last_sequence = sequence
            else:
                source_id = getattr(message, "id", None)
                if source_id is not None:
                    self.source_ids[int(source_id)] += 1
                nodes = list(getattr(message, "nodes", []))
                if not nodes:
                    self.empty_node_arrays += 1
                for node in nodes:
                    peer_id = getattr(node, "id", None)
                    if peer_id is not None:
                        self.peer_ids[int(peer_id)] += 1
                    if hasattr(node, "dis"):
                        self.range_count += 1
                        value = float(node.dis)
                        if not math.isfinite(value):
                            self.nonfinite_ranges += 1
                        elif value <= 0.0:
                            self.nonpositive_ranges += 1
                    self.has_fp_rssi = self.has_fp_rssi or hasattr(node, "fp_rssi")
                    self.has_rx_rssi = self.has_rx_rssi or hasattr(node, "rx_rssi")
                    self.has_cir_quality = self.has_cir_quality or hasattr(
                        node, "cir_quality"
                    )
                for field in ("local_time", "system_time"):
                    value = getattr(message, field, None)
                    if value is not None:
                        first_name = field + "_first"
                        last_name = field + "_last"
                        if getattr(self, first_name) is None:
                            setattr(self, first_name, int(value))
                        setattr(self, last_name, int(value))

    def report(self):
        with self.lock:
            report = {
                "message_count": len(self.receive_times),
                "receive_rate_hz": _rate(self.receive_times),
                "receive_intervals": _interval_summary(self.receive_times),
            }
            if self.role.startswith("mocap"):
                report.update(
                    {
                        "valid_pose_count": len(self.header_times),
                        "header_rate_hz": _rate(self.header_times),
                        "header_intervals": _interval_summary(self.header_times),
                        "zero_header_stamp_count": self.zero_header_stamp,
                        "sequence_missing_count": self.sequence_missing,
                        "frame_ids": dict(sorted(self.frame_ids.items())),
                        "invalid_pose_reasons": dict(self.invalid_pose),
                        "quaternion_norm_min": min(self.quaternion_norms)
                        if self.quaternion_norms
                        else None,
                        "quaternion_norm_max": max(self.quaternion_norms)
                        if self.quaternion_norms
                        else None,
                        "header_delay_min_s": min(self.header_delays)
                        if self.header_delays
                        else None,
                        "header_delay_max_s": max(self.header_delays)
                        if self.header_delays
                        else None,
                    }
                )
            else:
                report.update(
                    {
                        "source_ids": dict(sorted(self.source_ids.items())),
                        "peer_ids": dict(sorted(self.peer_ids.items())),
                        "empty_node_array_count": self.empty_node_arrays,
                        "range_count": self.range_count,
                        "nonfinite_range_count": self.nonfinite_ranges,
                        "nonpositive_range_count": self.nonpositive_ranges,
                        "has_fp_rssi": self.has_fp_rssi,
                        "has_rx_rssi": self.has_rx_rssi,
                        "has_cir_quality": self.has_cir_quality,
                        "local_time_first": self.local_time_first,
                        "local_time_last": self.local_time_last,
                        "system_time_first": self.system_time_first,
                        "system_time_last": self.system_time_last,
                    }
                )
            return report


def inspect(config, duration_s):
    import rosgraph
    import rospy

    rospy.init_node("uwb_bias_interface_inspector", anonymous=True, disable_signals=True)
    master = rosgraph.Master(rospy.get_name())
    published = dict(master.getPublishedTopics("/"))
    try:
        use_sim_time = bool(rospy.get_param("/use_sim_time"))
        use_sim_time_set = True
    except KeyError:
        use_sim_time = False
        use_sim_time_set = False

    report = {
        "inspection_started_unix_s": time.time(),
        "duration_s": float(duration_s),
        "use_sim_time": use_sim_time,
        "use_sim_time_parameter_set": use_sim_time_set,
        "topics": {},
    }
    subscribers = []
    probes = {}
    roles = ("uwb", "mocap_a", "mocap_b")
    for role in roles:
        item = config[role]
        topic = item.get("topic")
        expected_type = item.get("message_type")
        topic_report = {
            "configured_topic": topic,
            "configured_type": expected_type,
            "published": topic in published,
            "published_type": published.get(topic),
            "type_matches": published.get(topic) == expected_type if topic else False,
        }
        report["topics"][role] = topic_report
        if not topic or topic not in published:
            continue
        actual_type = published[topic]
        try:
            message_class = load_ros_message_class(actual_type)
        except RosAdapterError as exc:
            topic_report["class_load_error"] = str(exc)
            continue
        probe = TopicProbe(role, expected_type)
        probes[role] = probe
        subscribers.append(
            rospy.Subscriber(topic, message_class, probe.callback, queue_size=1000)
        )

    deadline = time.monotonic() + max(0.0, float(duration_s))
    while time.monotonic() < deadline and not rospy.is_shutdown():
        time.sleep(min(0.05, deadline - time.monotonic()))

    for subscriber in subscribers:
        subscriber.unregister()
    for role, probe in probes.items():
        report["topics"][role]["sample"] = probe.report()

    report["all_topics_published"] = all(
        report["topics"][role]["published"] for role in roles
    )
    report["all_types_match"] = all(
        report["topics"][role]["type_matches"] for role in roles
    )
    report["inspection_finished_unix_s"] = time.time()
    return report


def list_published_topics():
    """Return the live ROS topic/type table without requiring Gate 1 values."""

    import rosgraph
    import rospy

    rospy.init_node(
        "uwb_bias_topic_inventory", anonymous=True, disable_signals=True
    )
    master = rosgraph.Master(rospy.get_name())
    return {
        "published_topics": [
            {"topic": topic, "message_type": message_type}
            for topic, message_type in sorted(master.getPublishedTopics("/"))
        ]
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "ros_topics.yaml"),
    )
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--output-json")
    parser.add_argument(
        "--list-topics",
        action="store_true",
        help=(
            "List the live topic/type table without requiring resolved UWB IDs or "
            "final topic choices. This is a discovery action only."
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Print unresolved config without contacting ROS; exits with status 2.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if args.list_topics:
            report = list_published_topics()
            text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
            print(text)
            if args.output_json:
                write_json_exclusive(args.output_json, report)
            return 0
        config = load_yaml(args.config)
        validate_ros_topics_config(config, live=not args.allow_incomplete)
        if args.allow_incomplete:
            try:
                validate_ros_topics_config(config, live=True)
            except ConfigError as exc:
                payload = {"ready_for_live_inspection": False, "reason": str(exc)}
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return 2
        report = inspect(config, args.duration_s)
        text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        print(text)
        if args.output_json:
            write_json_exclusive(args.output_json, report)
        return 0 if report["all_topics_published"] and report["all_types_match"] else 3
    except (ConfigError, RosAdapterError) as exc:
        print("configuration/interface error: %s" % exc, file=sys.stderr)
        return 2
    except Exception as exc:
        print("ROS inspection failed: %s" % exc, file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
