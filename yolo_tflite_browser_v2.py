#!/usr/bin/env python3
import argparse
import os
import socket
import threading
import time
from typing import List, Tuple

import cv2
import numpy as np
from flask import Flask, Response

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter  # fallback
    from tensorflow.lite.python.interpreter import load_delegate


# COCO 80-class names used by YOLOv8n
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


def get_ip_addresses() -> List[Tuple[str, str]]:
    results = [("localhost", "127.0.0.1")]
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip != "127.0.0.1" and (hostname, ip) not in results:
                results.append((hostname, ip))
    except Exception:
        pass

    # Also try common interface lookup by connecting outward
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ("network", ip) not in results and ip != "127.0.0.1":
            results.append(("network", ip))
    except Exception:
        pass

    # Remove duplicates by IP
    seen = set()
    uniq = []
    for name, ip in results:
        if ip not in seen:
            uniq.append((name, ip))
            seen.add(ip)
    return uniq


def letterbox(
    image: np.ndarray,
    new_shape=(640, 640),
    color=(114, 114, 114)
):
    h, w = image.shape[:2]
    new_h, new_w = new_shape

    r = min(new_w / w, new_h / h)
    resized_w, resized_h = int(round(w * r)), int(round(h * r))

    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = new_w - resized_w
    pad_h = new_h - resized_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, r, left, top


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union

        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return keep


class YoloV8EdgeTPU:
    def __init__(self, model_path: str, conf_thres=0.25, iou_thres=0.45):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate("libedgetpu.so.1")]
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_index = self.input_details[0]["index"]
        self.output_index = self.output_details[0]["index"]

        input_shape = self.input_details[0]["shape"]
        # usually [1, H, W, 3]
        self.in_h = int(input_shape[1])
        self.in_w = int(input_shape[2])

        self.input_dtype = self.input_details[0]["dtype"]
        self.input_quant = self.input_details[0]["quantization"]  # (scale, zero_point)
        self.output_quant = self.output_details[0]["quantization"]

    def preprocess(self, frame: np.ndarray):
        img, ratio, pad_x, pad_y = letterbox(frame, (self.in_h, self.in_w))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.input_dtype == np.uint8:
            scale, zero = self.input_quant
            if scale == 0:
                inp = rgb.astype(np.uint8)
            else:
                inp = np.clip(np.round(rgb / scale + zero), 0, 255).astype(np.uint8)
        else:
            inp = rgb.astype(np.float32) / 255.0

        inp = np.expand_dims(inp, axis=0)
        return inp, ratio, pad_x, pad_y

    def infer(self, frame: np.ndarray):
        orig_h, orig_w = frame.shape[:2]
        inp, ratio, pad_x, pad_y = self.preprocess(frame)

        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)

        # dequantize if needed
        if self.output_details[0]["dtype"] == np.uint8:
            scale, zero = self.output_quant
            if scale != 0:
                out = (out.astype(np.float32) - zero) * scale
            else:
                out = out.astype(np.float32)

        boxes, scores, class_ids = self.decode_yolov8_output(out, orig_w, orig_h, ratio, pad_x, pad_y)
        return boxes, scores, class_ids

    def decode_yolov8_output(self, output, orig_w, orig_h, ratio, pad_x, pad_y):
        """
        Handles common YOLOv8 TFLite output layouts:
          [1, 84, 8400] or [1, 8400, 84]
        84 = 4 box values + 80 class scores
        """
        pred = np.squeeze(output)

        if pred.ndim != 2:
            raise RuntimeError(f"Unexpected output shape after squeeze: {pred.shape}")

        if pred.shape[0] in (84, 85) and pred.shape[1] > 100:
            pred = pred.T  # -> [N, C]
        elif pred.shape[1] in (84, 85):
            pass
        else:
            raise RuntimeError(f"Unsupported YOLOv8 output shape: {pred.shape}")

        # YOLOv8 export usually has no objectness in TFLite head: [x, y, w, h, cls...]
        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_scores)), class_ids]

        mask = scores >= self.conf_thres
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return np.empty((0, 4)), np.array([]), np.array([])

        # xywh -> xyxy on letterboxed input scale
        x = boxes_xywh[:, 0]
        y = boxes_xywh[:, 1]
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]

        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        # Undo letterbox
        x1 = (x1 - pad_x) / ratio
        y1 = (y1 - pad_y) / ratio
        x2 = (x2 - pad_x) / ratio
        y2 = (y2 - pad_y) / ratio

        x1 = np.clip(x1, 0, orig_w - 1)
        y1 = np.clip(y1, 0, orig_h - 1)
        x2 = np.clip(x2, 0, orig_w - 1)
        y2 = np.clip(y2, 0, orig_h - 1)

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # class-wise NMS
        final_boxes = []
        final_scores = []
        final_class_ids = []

        for cls in np.unique(class_ids):
            inds = np.where(class_ids == cls)[0]
            cls_boxes = boxes[inds]
            cls_scores = scores[inds]
            keep = nms(cls_boxes, cls_scores, self.iou_thres)

            final_boxes.append(cls_boxes[keep])
            final_scores.append(cls_scores[keep])
            final_class_ids.append(np.full(len(keep), cls, dtype=np.int32))

        if not final_boxes:
            return np.empty((0, 4)), np.array([]), np.array([])

        return (
            np.concatenate(final_boxes, axis=0),
            np.concatenate(final_scores, axis=0),
            np.concatenate(final_class_ids, axis=0),
        )


def draw_detections(frame, boxes, scores, class_ids, labels):
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cls_name = labels[int(cls_id)] if 0 <= int(cls_id) < len(labels) else str(int(cls_id))
        text = f"{cls_name} {score:.2f}"

        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(y1, th + 4)
        cv2.rectangle(frame, (x1, y_text - th - 4), (x1 + tw, y_text + baseline - 4), (0, 255, 0), -1)
        cv2.putText(
            frame, text, (x1, y_text - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )


class StreamState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_jpeg = None
        self.fps = 0.0
        self.running = True


def make_app(state: StreamState):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return """
        <html>
        <head>
            <title>YOLOv8 EdgeTPU Stream</title>
            <style>
                body { font-family: Arial, sans-serif; background: #111; color: #eee; text-align: center; }
                img { max-width: 95vw; max-height: 85vh; border: 2px solid #444; margin-top: 10px; }
            </style>
        </head>
        <body>
            <h2>YOLOv8 EdgeTPU USB Camera Stream</h2>
            <img src="/video_feed">
        </body>
        </html>
        """

    def gen():
        while True:
            with state.lock:
                frame = state.frame_jpeg
            if frame is None:
                time.sleep(0.01)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(0.01)

    @app.route("/video_feed")
    def video_feed():
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def camera_worker(args, state: StreamState):
    model = YoloV8EdgeTPU(
        model_path=args.model,
        conf_thres=args.conf,
        iou_thres=args.iou,
    )

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    prev_t = time.time()

    while state.running:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        boxes, scores, class_ids = model.infer(frame)
        draw_detections(frame, boxes, scores, class_ids, COCO80)

        now = time.time()
        dt = now - prev_t
        prev_t = now
        fps = 1.0 / dt if dt > 0 else 0.0
        state.fps = fps

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with state.lock:
                state.frame_jpeg = jpeg.tobytes()

    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n_full_integer_quant_edgetpu.tflite",
                        help="Path to EdgeTPU YOLOv8 TFLite model")
    parser.add_argument("--camera", type=int, default=1, help="USB camera index")
    parser.add_argument("--width", type=int, default=320, help="Camera width")
    parser.add_argument("--height", type=int, default=320, help="Camera height")
    parser.add_argument("--host", default="0.0.0.0", help="Flask bind host")
    parser.add_argument("--port", type=int, default=5000, help="Flask port")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    state = StreamState()

    t = threading.Thread(target=camera_worker, args=(args, state), daemon=True)
    t.start()

    print("\nOpen the stream in your browser:")
    for name, ip in get_ip_addresses():
        print(f"  {name:10s} http://{ip}:{args.port}")

    app = make_app(state)
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        state.running = False


if __name__ == "__main__":
    main()
