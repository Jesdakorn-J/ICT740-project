#!/usr/bin/env python3
"""Cross-platform USB/internal camera MJPEG server for browser viewing.

Works on Windows, macOS, and Linux.

Examples:
    python usb_cam_browserupdated_rewritten.py
    python usb_cam_browserupdated_rewritten.py --camera 0 --width 1280 --height 720 --port 8080
    python usb_cam_browserupdated_rewritten.py --list-cameras
"""

from __future__ import annotations

import argparse
import json
import platform
import signal
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Optional

import cv2


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>USB Camera Stream</title>
  <style>
    :root { color-scheme: dark; }
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background: #0f1115;
      color: #e7eaf0;
      text-align: center;
    }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 24px; }
    .panel {
      background: #171b22;
      border: 1px solid #2a3140;
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    img {
      width: 100%;
      max-width: 1000px;
      border-radius: 12px;
      background: black;
    }
    a {
      color: #8ec5ff;
      text-decoration: none;
      margin: 0 8px;
    }
    a:hover { text-decoration: underline; }
    .muted { color: #b8c0cc; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>USB Camera Browser Stream</h1>
    <p class="muted">Live MJPEG feed from the selected camera.</p>
    <div class="panel">
      <img src="/stream.mjpg" alt="Live camera stream">
      <p>
        <a href="/stream.mjpg">Direct stream</a>
        <a href="/status">Status</a>
      </p>
    </div>
  </div>
</body>
</html>
"""


@dataclass
class AppConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    camera_index: int = 0
    width: int = 640
    height: int = 480
    jpeg_quality: int = 80
    mirror: bool = True
    fps_limit: float = 30.0
    reconnect_delay: float = 1.0
    backend_name: str = "auto"


class CameraStream:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.lock = threading.Lock()
        self.running = True
        self.latest_jpeg: Optional[bytes] = None
        self.latest_frame_time: float = 0.0
        self.current_fps: float = 0.0
        self.frame_count: int = 0
        self.last_error: Optional[str] = None
        self.backend_used: str = "unknown"

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def _candidate_backends(self) -> list[tuple[str, Optional[int]]]:
        requested = self.config.backend_name.lower()
        candidates: list[tuple[str, Optional[int]]] = []

        backend_map = {
            "auto": [("auto", None)],
            "default": [("auto", None)],
            "dshow": [("dshow", getattr(cv2, "CAP_DSHOW", None))],
            "msmf": [("msmf", getattr(cv2, "CAP_MSMF", None))],
            "avfoundation": [("avfoundation", getattr(cv2, "CAP_AVFOUNDATION", None))],
            "v4l2": [("v4l2", getattr(cv2, "CAP_V4L2", None))],
        }

        if requested in backend_map:
            return backend_map[requested]

        system_name = platform.system().lower()
        if system_name == "windows":
            candidates.extend([
                ("auto", None),
                ("dshow", getattr(cv2, "CAP_DSHOW", None)),
                ("msmf", getattr(cv2, "CAP_MSMF", None)),
            ])
        elif system_name == "darwin":
            candidates.extend([
                ("auto", None),
                ("avfoundation", getattr(cv2, "CAP_AVFOUNDATION", None)),
            ])
        else:
            candidates.extend([
                ("auto", None),
                ("v4l2", getattr(cv2, "CAP_V4L2", None)),
            ])
        return candidates

    def _open_capture(self, backend_id: Optional[int]) -> cv2.VideoCapture:
        if backend_id is None:
            cap = cv2.VideoCapture(self.config.camera_index)
        else:
            cap = cv2.VideoCapture(self.config.camera_index, backend_id)
        return cap

    def _open_camera(self) -> cv2.VideoCapture:
        errors: list[str] = []

        for backend_name, backend_id in self._candidate_backends():
            try:
                cap = self._open_capture(backend_id)
                if not cap or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    errors.append(f"{backend_name}: not opened")
                    continue

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.release()
                    errors.append(f"{backend_name}: opened but no frame")
                    continue

                self.backend_used = backend_name
                return cap
            except Exception as exc:
                errors.append(f"{backend_name}: {exc}")

        raise RuntimeError(
            f"Could not open camera index {self.config.camera_index}. Tried backends: "
            + "; ".join(errors)
        )

    def _reset_camera(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

    def _capture_loop(self) -> None:
        previous_ts = time.time()
        frame_interval = 1.0 / self.config.fps_limit if self.config.fps_limit > 0 else 0.0

        while self.running:
            try:
                if self.cap is None:
                    self.cap = self._open_camera()
                    self.last_error = None

                ok, frame = self.cap.read()
                if not ok or frame is None:
                    self.last_error = "Camera read failed. Reconnecting..."
                    self._reset_camera()
                    time.sleep(self.config.reconnect_delay)
                    continue

                if self.config.mirror:
                    frame = cv2.flip(frame, 1)

                now = time.time()
                dt = max(now - previous_ts, 1e-6)
                previous_ts = now
                self.current_fps = 1.0 / dt
                self.frame_count += 1
                self.latest_frame_time = now

                overlay = (
                    f"Cam {self.config.camera_index} | {self.backend_used} | "
                    f"FPS {self.current_fps:.1f} | Frames {self.frame_count}"
                )
                cv2.putText(
                    frame,
                    overlay,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality],
                )
                if ok:
                    with self.lock:
                        self.latest_jpeg = encoded.tobytes()

                if frame_interval > 0:
                    sleep_time = frame_interval - (time.time() - now)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            except Exception as exc:
                self.last_error = str(exc)
                self._reset_camera()
                time.sleep(self.config.reconnect_delay)

    def get_jpeg_frame(self) -> Optional[bytes]:
        with self.lock:
            return self.latest_jpeg

    def get_status(self) -> dict:
        return {
            "running": self.running,
            "camera_index": self.config.camera_index,
            "width": self.config.width,
            "height": self.config.height,
            "jpeg_quality": self.config.jpeg_quality,
            "mirror": self.config.mirror,
            "fps": round(self.current_fps, 2),
            "frame_count": self.frame_count,
            "last_frame_unix": self.latest_frame_time,
            "last_error": self.last_error,
            "camera_open": self.cap is not None and self.cap.isOpened(),
            "backend_used": self.backend_used,
        }

    def stop(self) -> None:
        self.running = False
        self._capture_thread.join(timeout=2.0)
        self._reset_camera()


class StreamHandler(BaseHTTPRequestHandler):
    server_version = "USBCamBrowser/2.0"

    def do_GET(self) -> None:
        if self.path == "/":
            payload = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path == "/status":
            payload = json.dumps(self.server.camera.get_status(), indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path == "/stream.mjpg":
            self._handle_mjpeg_stream()
            return

        self.send_error(404, "Not found")

    def _handle_mjpeg_stream(self) -> None:
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            while True:
                frame = self.server.camera.get_jpeg_frame()
                if frame is None:
                    time.sleep(0.03)
                    continue

                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(0.001)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def log_message(self, fmt: str, *args) -> None:
        return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def find_available_cameras(max_index: int = 10) -> list[int]:
    found: list[int] = []
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index)
        try:
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            if ok and frame is not None:
                found.append(index)
        finally:
            cap.release()
    return found


def parse_args() -> tuple[AppConfig, bool]:
    parser = argparse.ArgumentParser(description="Serve a USB or internal camera stream to a web browser.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--camera", type=int, default=0, dest="camera_index")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--fps", type=float, default=30.0, dest="fps_limit")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument(
        "--backend",
        choices=["auto", "default", "dshow", "msmf", "avfoundation", "v4l2"],
        default="auto",
        help="Camera backend to try first. Use auto unless you need to force one.",
    )
    parser.add_argument("--list-cameras", action="store_true", help="Probe and print working camera indexes, then exit.")
    args = parser.parse_args()

    config = AppConfig(
        host=args.host,
        port=args.port,
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        jpeg_quality=max(30, min(95, args.jpeg_quality)),
        mirror=not args.no_mirror,
        fps_limit=max(1.0, args.fps_limit),
        backend_name=args.backend,
    )
    return config, args.list_cameras


def run_server(config: AppConfig) -> int:
    camera = CameraStream(config)
    server = ThreadedHTTPServer((config.host, config.port), StreamHandler)
    server.camera = camera

    def handle_shutdown(signum=None, frame=None) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, handle_shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_shutdown)

    print("USB camera server started")
    print(f"System       : {platform.system()}")
    print(f"Camera index : {config.camera_index}")
    print(f"Resolution   : {config.width}x{config.height}")
    print(f"Backend pref : {config.backend_name}")
    print(f"Open browser : http://127.0.0.1:{config.port}")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    finally:
        server.server_close()
        camera.stop()
        print("Server stopped")

    return 0


def main() -> int:
    config, list_cameras = parse_args()

    if list_cameras:
        found = find_available_cameras()
        if found:
            print("Working camera indexes:", ", ".join(str(i) for i in found))
        else:
            print("No working cameras found. Check the camera cable, permissions, or try another app.")
        return 0

    return run_server(config)


if __name__ == "__main__":
    sys.exit(main())
