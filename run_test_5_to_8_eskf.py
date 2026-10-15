#!/usr/bin/env python3
"""Replay test_5..test_8 with both ESKF executables in isolated ROS sessions."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import xmlrpc.client


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "eskf_test_5_to_8_results"
LAUNCH = ROOT / "src/uwb_pose_estimation/launch/uwb_imu_tof_eskf_real.launch"
NODES = (
    "uwb_imu_tof_eskf_real_add_imu_bias_node",
    "uwb_imu_tof_eskf_real_node",
)


def start(command, log_path, env):
    log_file = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=log_file,
        stderr=subprocess.STDOUT, start_new_session=True,
    )
    return process, log_file


def stop(process_and_log):
    if process_and_log is None:
        return
    process, log_file = process_and_log
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
    log_file.close()


def wait_for_master(uri, process, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("roscore exited; inspect roscore.log")
        try:
            xmlrpc.client.ServerProxy(uri).getUri("/eskf_replay")
            return
        except (OSError, xmlrpc.client.Error):
            time.sleep(0.3)
    raise RuntimeError("ROS master did not become ready")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=11360)
    parser.add_argument("--bags", nargs="*", type=int, default=list(range(5, 9)))
    parser.add_argument("--nodes", nargs="*", choices=NODES, default=list(NODES))
    parser.add_argument("--fixed-zero-bias", action="store_true",
                        help="Disable online estimation in the bias executable")
    parser.add_argument("--record-bias", action="store_true",
                        help="Record four bias diagnostic topics")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    uri = f"http://localhost:{args.port}"
    env = os.environ.copy()
    env.update(ROS_MASTER_URI=uri, ROS_HOSTNAME="localhost",
               ROS_HOME=str(RESULTS / "ros_home"))
    master = start(["roscore", "-p", str(args.port)], RESULTS / "roscore.log", env)
    try:
        wait_for_master(uri, master[0])
        subprocess.run(["rosparam", "set", "/use_sim_time", "true"],
                       env=env, cwd=ROOT, check=True)
        original_launch = LAUNCH.read_text(encoding="utf-8")
        for bag in args.bags:
            bag_path = ROOT / f"test_{bag}.bag"
            if not bag_path.is_file():
                raise FileNotFoundError(bag_path)
            for node in args.nodes:
                variant = "_fixed_zero_bias" if args.fixed_zero_bias else ""
                prefix = f"test_{bag}_{node}{variant}"
                csv_path = RESULTS / f"{prefix}.csv"
                launch_path = RESULTS / f"{prefix}.launch"
                launch_text = original_launch.replace(
                        "/home/yu/catkin_workspace/eskf_pose_compare_real.csv",
                        str(csv_path))
                if args.fixed_zero_bias:
                    launch_text = launch_text.replace(
                        '<param name="publish_rate_hz" value="50.0"/>',
                        '<param name="publish_rate_hz" value="50.0"/>\n'
                        '    <param name="estimate_imu_bias" value="false"/>')
                launch_path.write_text(launch_text, encoding="utf-8")
                ground = eskf = recorder = None
                print(f"START {prefix}", flush=True)
                try:
                    ground = start(["roslaunch", "imu_mahony", "ground_distance.launch"],
                                   RESULTS / f"{prefix}_ground.log", env)
                    eskf = start(["roslaunch", str(launch_path),
                                  f"eskf_node_type:={node}"],
                                 RESULTS / f"{prefix}_eskf.log", env)
                    time.sleep(3)
                    if ground[0].poll() is not None or eskf[0].poll() is not None:
                        raise RuntimeError(f"launch exited early: {prefix}")
                    if args.record_bias:
                        topics = [f"/uwb_imu_tof_eskf_real_node/bias/{kind}_{body}"
                                  for kind in ("gyro", "acc") for body in ("A", "B")]
                        recorder = start(["rosbag", "record", "-O",
                                          str(RESULTS / f"{prefix}_bias.bag"), *topics],
                                         RESULTS / f"{prefix}_record.log", env)
                        time.sleep(1)
                    with open(RESULTS / f"{prefix}_bag.log", "w", encoding="utf-8") as bag_log:
                        subprocess.run(["rosbag", "play", "--clock", "--delay=2",
                                        str(bag_path)], cwd=ROOT, env=env,
                                       stdout=bag_log, stderr=subprocess.STDOUT,
                                       check=True)
                    time.sleep(1)
                finally:
                    stop(recorder)
                    stop(eskf)
                    stop(ground)
                if not csv_path.is_file() or csv_path.stat().st_size < 1000:
                    raise RuntimeError(f"No useful CSV data: {csv_path}")
                print(f"DONE {prefix}: {csv_path.stat().st_size} bytes", flush=True)
    finally:
        stop(master)


if __name__ == "__main__":
    main()
