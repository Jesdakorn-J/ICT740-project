#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import threading
import time
import cv2


HOST = "0.0.0.0"
PORT = 8080
CAMERA_INDEX = 1       # /dev/video0 -> 0, /dev/video1 -> 1
WIDTH = 640
HEIGHT = 480
JPEG_QUALITY = 80


class CameraStream:
    def __init__(self, camera_index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open USB camera.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.lock = threading.Lock()
        self.frame_jpeg = None
        self.running = True
        self.fps = 0.0

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        prev_time = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # Optional mirror view
            frame = cv2.flip(frame, 1)

            # FPS
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                self.fps = 1.0 / dt

            # Overlay text
            cv2.putText(
                frame,
                f"Coral USB Camera | FPS: {self.fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            ok, jpeg = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if ok:
                with self.lock:
                    self.frame_jpeg = jpeg.tobytes()

    def get_frame(self):
        with self.lock:
            return self.frame_jpeg

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


PAGE = b"""\
<html>
<head>
    <title>Coral USB Camera Stream</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #111;
            color: #eee;
            text-align: center;
            margin: 0;
            padding: 20px;
        }
        img {
            max-width: 95vw;
            max-height: 85vh;
            border: 3px solid #444;
            border-radius: 8px;
        }
        .box {
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <h2>Coral USB Camera Stream</h2>
    <div class="box">
        <img src="/stream.mjpg" />
    </div>
</body>
</html>
"""


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)

        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            try:
                while True:
                    frame = self.server.camera.get_frame()
                    if frame is None:
                        time.sleep(0.03)
                        continue

                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                pass

        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, format, *args):
        return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    print("Opening camera...")
    camera = CameraStream(
        camera_index=CAMERA_INDEX,
        width=WIDTH,
        height=HEIGHT
    )

    server = ThreadedHTTPServer((HOST, PORT), StreamHandler)
    server.camera = camera

    print(f"Stream started.")
    print(f"Open in browser: http://<CORAL_BOARD_IP>:{PORT}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        server.server_close()
        camera.stop()


if __name__ == "__main__":
    main()
