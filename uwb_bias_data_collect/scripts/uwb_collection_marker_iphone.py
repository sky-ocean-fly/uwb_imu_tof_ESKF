#!/usr/bin/env python3
"""Publish UWB data-collection markers from a phone or the local keyboard.

Phone control:
    Open http://<computer-lan-ip>:8080 in a phone browser.

Keyboard fallback:
    A/a: start a static segment
    B/b: end the static segment and advance pose_id
    Q/q: quit

The marker is published as JSON in std_msgs/String on
``/uwb_bias/collection_marker``.  The rosbag message time remains the
authoritative timestamp for offline segmentation.
"""

import json
import select
import sys
import termios
import threading
import tty
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rospy
from std_msgs.msg import String


MARKER_TOPIC = "/uwb_bias/collection_marker"


CONTROL_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport"
        content="width=device-width, initial-scale=1, viewport-fit=cover">
  <title>UWB 数据采集标记</title>
  <style>
    :root {
      color-scheme: light;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
                   sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: #f3f5f8;
      color: #172033;
    }
    main {
      width: min(680px, 100%);
      margin: 0 auto;
      padding: 20px 16px 36px;
    }
    h1 {
      margin: 4px 0 18px;
      font-size: 25px;
      text-align: center;
    }
    .card {
      margin-bottom: 14px;
      padding: 18px;
      border-radius: 18px;
      background: white;
      box-shadow: 0 6px 24px rgba(25, 38, 64, 0.08);
    }
    .pose {
      margin: 4px 0 8px;
      font-size: 36px;
      font-weight: 750;
      text-align: center;
      letter-spacing: 0.02em;
    }
    .status {
      min-height: 28px;
      font-size: 18px;
      font-weight: 650;
      text-align: center;
    }
    .status.recording { color: #087f5b; }
    .status.waiting { color: #495057; }
    .connection {
      margin-top: 8px;
      text-align: center;
      font-size: 14px;
      color: #687386;
    }
    .connection.bad { color: #c92a2a; }
    .actions {
      display: grid;
      gap: 14px;
    }
    button {
      width: 100%;
      min-height: 92px;
      border: 0;
      border-radius: 18px;
      color: white;
      font-size: 25px;
      font-weight: 750;
      touch-action: manipulation;
      transition: transform 0.08s ease, opacity 0.2s ease;
    }
    button:active:not(:disabled) { transform: scale(0.985); }
    button:disabled { opacity: 0.34; }
    #startButton { background: #12a36d; }
    #moveButton { background: #f08c00; }
    .pose-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: center;
    }
    input {
      min-width: 0;
      height: 48px;
      padding: 0 12px;
      border: 1px solid #cfd6df;
      border-radius: 12px;
      font-size: 18px;
    }
    #setPoseButton {
      width: auto;
      min-height: 48px;
      padding: 0 18px;
      border-radius: 12px;
      background: #4263eb;
      font-size: 17px;
    }
    .message {
      min-height: 24px;
      margin-top: 12px;
      text-align: center;
      color: #42526b;
    }
    .message.error { color: #c92a2a; }
    .details {
      margin-top: 10px;
      font-size: 13px;
      color: #7a8495;
      text-align: center;
    }
  </style>
</head>
<body>
<main>
  <h1>UWB 数据采集标记</h1>

  <section class="card">
    <div id="poseLabel" class="pose">pose_---</div>
    <div id="statusLabel" class="status waiting">正在连接……</div>
    <div id="connectionLabel" class="connection">检查 ROS 订阅者</div>
  </section>

  <section class="card actions">
    <button id="startButton" onclick="sendAction('/api/static-start')">
      A · 开始静止
    </button>
    <button id="moveButton" onclick="sendAction('/api/move-start')">
      B · 准备移动
    </button>
  </section>

  <section class="card">
    <div class="pose-row">
      <input id="poseInput" type="number" min="0" step="1"
             inputmode="numeric" placeholder="设置 pose_id">
      <button id="setPoseButton" onclick="setPose()">设置编号</button>
    </div>
    <div id="messageLabel" class="message"></div>
    <div id="lastEventLabel" class="details">尚未发布标记</div>
  </section>
</main>

<script>
  let busy = false;
  let latestState = null;

  function poseName(id) {
    return `pose_${String(id).padStart(3, "0")}`;
  }

  function showMessage(text, isError = false) {
    const label = document.getElementById("messageLabel");
    label.textContent = text || "";
    label.className = isError ? "message error" : "message";
  }

  function render(state) {
    latestState = state;
    const connected = state.subscriber_count > 0;

    document.getElementById("poseLabel").textContent =
      poseName(state.pose_id);

    const status = document.getElementById("statusLabel");
    if (state.is_static) {
      status.textContent = "正在记录静止数据";
      status.className = "status recording";
    } else {
      status.textContent = "等待开始静止";
      status.className = "status waiting";
    }

    const connection = document.getElementById("connectionLabel");
    if (connected) {
      connection.textContent =
        `标记话题已连接（订阅者 ${state.subscriber_count}）`;
      connection.className = "connection";
    } else {
      connection.textContent = "没有订阅者，请先启动 rosbag record";
      connection.className = "connection bad";
    }

    document.getElementById("startButton").disabled =
      busy || state.is_static || !connected;
    document.getElementById("moveButton").disabled =
      busy || !state.is_static || !connected;
    document.getElementById("setPoseButton").disabled =
      busy || state.is_static;
    document.getElementById("poseInput").disabled =
      busy || state.is_static;

    const last = state.last_event;
    document.getElementById("lastEventLabel").textContent = last
      ? `最近标记：${last.event} · ${poseName(last.pose_id)}`
      : "尚未发布标记";
  }

  async function getState() {
    try {
      const response = await fetch("/api/state", {cache: "no-store"});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      render(await response.json());
    } catch (error) {
      showMessage(`无法连接控制节点：${error.message}`, true);
    }
  }

  async function sendAction(path, body = {}) {
    if (busy) return;
    busy = true;
    if (latestState) render(latestState);

    try {
      const response = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
      });
      const result = await response.json();
      showMessage(result.message, !result.ok);
      render(result);
      if (result.ok && navigator.vibrate) navigator.vibrate(80);
    } catch (error) {
      showMessage(`操作失败：${error.message}`, true);
    } finally {
      busy = false;
      if (latestState) render(latestState);
      await getState();
    }
  }

  function setPose() {
    const value = Number(document.getElementById("poseInput").value);
    if (!Number.isInteger(value) || value < 0) {
      showMessage("请输入非负整数 pose_id。", true);
      return;
    }
    sendAction("/api/set-pose", {pose_id: value});
  }

  getState();
  setInterval(getState, 1000);
</script>
</body>
</html>
"""


class MarkerController:
    """Own the marker state machine shared by phone and keyboard controls."""

    def __init__(self, start_pose_id, require_subscriber=True):
        self.pose_id = start_pose_id
        self.is_static = False
        self.last_event = None
        self.require_subscriber = require_subscriber
        self.lock = threading.Lock()
        self.publisher = rospy.Publisher(
            MARKER_TOPIC,
            String,
            queue_size=10,
            latch=False,
        )

    def _snapshot_unlocked(self):
        return {
            "pose_id": self.pose_id,
            "is_static": self.is_static,
            "subscriber_count": self.publisher.get_num_connections(),
            "last_event": self.last_event,
            "topic": MARKER_TOPIC,
        }

    def _reply_unlocked(self, ok, message):
        result = self._snapshot_unlocked()
        result.update({"ok": ok, "message": message})
        return result

    def state(self):
        with self.lock:
            return self._snapshot_unlocked()

    def _publish_unlocked(self, event, source):
        subscriber_count = self.publisher.get_num_connections()
        if self.require_subscriber and subscriber_count == 0:
            return False, "没有标记话题订阅者，请先启动 rosbag record。"

        stamp = rospy.Time.now().to_sec()
        marker = {
            "event": event,
            "pose_id": self.pose_id,
            "stamp": stamp,
            "source": source,
        }
        self.publisher.publish(
            String(data=json.dumps(marker, separators=(",", ":")))
        )
        self.last_event = marker

        rospy.loginfo(
            "已发布 %s，pose_%03d，来源=%s",
            event,
            self.pose_id,
            source,
        )
        return True, ""

    def start_static(self, source):
        with self.lock:
            if self.is_static:
                return self._reply_unlocked(
                    False,
                    "当前已经处于静止记录状态，未重复发布。",
                )

            ok, error = self._publish_unlocked("STATIC_START", source)
            if not ok:
                return self._reply_unlocked(False, error)

            self.is_static = True
            return self._reply_unlocked(
                True,
                "已记录开始静止：pose_%03d" % self.pose_id,
            )

    def start_moving(self, source):
        with self.lock:
            if not self.is_static:
                return self._reply_unlocked(
                    False,
                    "当前没有正在记录的静止段，请先按“开始静止”。",
                )

            finished_pose_id = self.pose_id
            ok, error = self._publish_unlocked("MOVE_START", source)
            if not ok:
                return self._reply_unlocked(False, error)

            self.is_static = False
            self.pose_id += 1
            return self._reply_unlocked(
                True,
                "pose_%03d 已结束，下一个是 pose_%03d"
                % (finished_pose_id, self.pose_id),
            )

    def set_pose_id(self, pose_id):
        with self.lock:
            if self.is_static:
                return self._reply_unlocked(
                    False,
                    "静止段正在记录，不能修改 pose_id。",
                )
            if pose_id < 0:
                return self._reply_unlocked(
                    False,
                    "pose_id 必须是非负整数。",
                )

            self.pose_id = pose_id
            rospy.loginfo("当前位姿编号已设置为 pose_%03d", self.pose_id)
            return self._reply_unlocked(
                True,
                "当前位姿已设置为 pose_%03d" % self.pose_id,
            )


class MarkerRequestHandler(BaseHTTPRequestHandler):
    """Small same-origin HTTP API and mobile control page."""

    server_version = "UwbMarker/1.0"

    @property
    def controller(self):
        return self.server.controller

    def _send_bytes(self, body, content_type, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self._send_bytes(
            body,
            "application/json; charset=utf-8",
            status,
        )

    def _read_json(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("无效的 Content-Length。") from exc

        if content_length < 0 or content_length > 4096:
            raise ValueError("请求内容过大。")
        if content_length == 0:
            return {}

        raw = self.rfile.read(content_length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("请求不是有效的 JSON。") from exc

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send_bytes(
                CONTROL_PAGE.encode("utf-8"),
                "text/html; charset=utf-8",
            )
        elif path == "/api/state":
            self._send_json(self.controller.state())
        else:
            self._send_json(
                {"ok": False, "message": "页面不存在。"},
                HTTPStatus.NOT_FOUND,
            )

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            body = self._read_json()
        except ValueError as exc:
            self._send_json(
                {"ok": False, "message": str(exc)},
                HTTPStatus.BAD_REQUEST,
            )
            return

        if path == "/api/static-start":
            result = self.controller.start_static("phone")
        elif path == "/api/move-start":
            result = self.controller.start_moving("phone")
        elif path == "/api/set-pose":
            pose_id = body.get("pose_id")
            if isinstance(pose_id, bool) or not isinstance(pose_id, int):
                result = self.controller.state()
                result.update(
                    {
                        "ok": False,
                        "message": "pose_id 必须是非负整数。",
                    }
                )
            else:
                result = self.controller.set_pose_id(pose_id)
        else:
            self._send_json(
                {"ok": False, "message": "接口不存在。"},
                HTTPStatus.NOT_FOUND,
            )
            return

        status = HTTPStatus.OK if result["ok"] else HTTPStatus.CONFLICT
        self._send_json(result, status)

    def log_message(self, _format, *_args):
        # Avoid one terminal log line for every one-second state refresh.
        return


class MarkerHttpServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def keyboard_loop(controller):
    """Retain the original one-key A/B control as a local fallback."""
    if not sys.stdin.isatty():
        rospy.logwarn("当前没有交互式终端，键盘 A/B 控制已禁用。")
        return

    terminal_fd = sys.stdin.fileno()
    try:
        old_settings = termios.tcgetattr(terminal_fd)
        tty.setcbreak(terminal_fd)
    except termios.error as exc:
        rospy.logwarn("无法启用键盘控制：%s", exc)
        return

    rospy.loginfo("键盘备用控制：A=开始静止，B=准备移动，Q=退出")
    try:
        while not rospy.is_shutdown():
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not readable:
                continue

            key = sys.stdin.read(1).lower()
            if key == "a":
                result = controller.start_static("keyboard")
            elif key == "b":
                result = controller.start_moving("keyboard")
            elif key == "q":
                if controller.state()["is_static"]:
                    rospy.logwarn("当前静止段尚未按 B 结束。")
                rospy.signal_shutdown("用户按下 Q")
                break
            else:
                continue

            if not result["ok"]:
                rospy.logwarn("%s", result["message"])
    finally:
        termios.tcsetattr(
            terminal_fd,
            termios.TCSADRAIN,
            old_settings,
        )


def main():
    rospy.init_node("uwb_collection_marker")

    start_pose_id = int(rospy.get_param("~start_pose_id", 1))
    port = int(rospy.get_param("~port", 8080))
    bind_host = str(rospy.get_param("~bind_host", "0.0.0.0"))
    require_subscriber = bool(
        rospy.get_param("~require_subscriber", True)
    )

    if start_pose_id < 0:
        rospy.logfatal("~start_pose_id 必须是非负整数。")
        return 2
    if not 1 <= port <= 65535:
        rospy.logfatal("~port 必须在 1 到 65535 之间。")
        return 2

    controller = MarkerController(
        start_pose_id=start_pose_id,
        require_subscriber=require_subscriber,
    )

    try:
        server = MarkerHttpServer(
            (bind_host, port),
            MarkerRequestHandler,
        )
    except OSError as exc:
        rospy.logfatal("无法启动手机控制网页：%s", exc)
        return 1

    server.controller = controller
    server_thread = threading.Thread(
        target=server.serve_forever,
        name="uwb-marker-http",
        daemon=True,
    )
    server_thread.start()

    keyboard_thread = threading.Thread(
        target=keyboard_loop,
        args=(controller,),
        name="uwb-marker-keyboard",
        daemon=True,
    )
    keyboard_thread.start()

    rospy.loginfo(
        "手机控制网页已启动："
        "请在同一局域网的手机上打开 http://<电脑局域网IP>:%d",
        port,
    )

    try:
        rospy.spin()
    finally:
        server.shutdown()
        server.server_close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
