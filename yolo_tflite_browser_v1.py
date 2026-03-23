#!/usr/bin/env python3
import os
import cv2
import time
import socket
import argparse
import threading
import numpy as np
from flask import Flask, Response

# Try lightweight runtime first (common on Coral / embedded Linux)
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
    load_delegate = None

# -----------------------------
# Config defaults
# -----------------------------
DEFAULT_MODEL = "yolov8n.tflite"
DEFAULT_LABELS = None
DEFAULT_CAMERA = 0
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000
DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.45
DEFAULT_IMG_SIZE = 640

app = Flask(__name__)

latest_jpeg = None
jpeg_lock = threading.Lock()


# -----------------------------
# Utilities
# -----------------------------
def load_labels(label_path):
    if not label_path or not os.path.exists(label_path):
        return None

    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # support either: "0 person" or just "person"
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1])
            else:
                labels.append(line)
    return labels


def get_ip_for_iface(iface_name):
    """
    Returns best-effort IP for a named interface like wlan0 or eth0.
    Falls back to None if unavailable.
    """
    try:
        import fcntl
        import struct

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return socket.inet_ntoa(
            fcntl.ioctl(
                s.fileno(),
                0x8915,  # SIOCGIFADDR
                struct.pack("256s", iface_name[:15].encode("utf-8")),
            )[20:24]
        )
    except Exception:
        return None


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize with unchanged aspect ratio using padding.
    Returns:
      image_resized, ratio, (dw, dh)
    """
    h, w = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return image, r, (dw, dh)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_iou(box, boxes):
    """
    box: [x1, y1, x2, y2]
    boxes: Nx4
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter = inter_w * inter_h

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = box_area + boxes_area - inter + 1e-6
    return inter / union


def nms(boxes, scores, iou_threshold=0.45):
    """
    Pure numpy NMS
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        ious = compute_iou(boxes[i], boxes[order[1:]])
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def xywh_to_xyxy(x, y, w, h):
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def scale_boxes_to_original(boxes, ratio, dwdh, orig_shape):
    """
    Undo letterbox transform
    """
    dw, dh = dwdh
    orig_h, orig_w = orig_shape[:2]
    scaled = []

    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio

        x1 = max(0, min(orig_w - 1, x1))
        y1 = max(0, min(orig_h - 1, y1))
        x2 = max(0, min(orig_w - 1, x2))
        y2 = max(0, min(orig_h - 1, y2))
        scaled.append([x1, y1, x2, y2])

    return scaled


# -----------------------------
# YOLOv8 TFLite wrapper
# -----------------------------
class YOLOv8TFLite:
    def __init__(self, model_path, conf_thres=0.35, iou_thres=0.45, use_edgetpu=False):
        delegates = None

        if use_edgetpu:
            if load_delegate is None:
                raise RuntimeError("EdgeTPU delegate requested, but load_delegate is unavailable.")
            delegates = [load_delegate("libedgetpu.so.1")]

        self.interpreter = Interpreter(model_path=model_path, experimental_delegates=delegates)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_index = self.input_details[0]["index"]
        self.output_index = self.output_details[0]["index"]

        self.input_shape = self.input_details[0]["shape"]  # e.g. [1, 640, 640, 3]
        self.input_dtype = self.input_details[0]["dtype"]
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.in_h = int(self.input_shape[1])
        self.in_w = int(self.input_shape[2])

        self.quant_in = "quantization" in self.input_details[0] and self.input_details[0]["quantization"] != (0.0, 0)
        self.quant_out = "quantization" in self.output_details[0] and self.output_details[0]["quantization"] != (0.0, 0)

    def preprocess(self, frame):
        img, ratio, dwdh = letterbox(frame, (self.in_h, self.in_w))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.input_dtype == np.float32:
            input_tensor = img_rgb.astype(np.float32) / 255.0
        else:
            input_tensor = img_rgb.astype(self.input_dtype)

        input_tensor = np.expand_dims(input_tensor, axis=0)

        if self.quant_in:
            scale, zero_point = self.input_details[0]["quantization"]
            if scale > 0:
                input_tensor = (input_tensor / scale + zero_point).astype(self.input_dtype)

        return input_tensor, ratio, dwdh

    def infer(self, frame):
        input_tensor, ratio, dwdh = self.preprocess(frame)

        self.interpreter.set_tensor(self.input_index, input_tensor)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_index)

        if self.quant_out:
            scale, zero_point = self.output_details[0]["quantization"]
            if scale > 0:
                output = (output.astype(np.float32) - zero_point) * scale

        detections = self.postprocess(output, frame.shape, ratio, dwdh)
        return detections

    def postprocess(self, output, orig_shape, ratio, dwdh):
        """
        Handles common YOLOv8 TFLite output layouts:
          - [1, 84, 8400]
          - [1, 8400, 84]

        Format assumed:
          [cx, cy, w, h, class_probs...]

        If your export includes objectness separately, you may need to adjust this section.
        """
        pred = np.squeeze(output)

        # Normalize shape to [num_boxes, num_features]
        if pred.ndim != 2:
            raise RuntimeError(f"Unexpected output shape after squeeze: {pred.shape}")

        # Usually YOLOv8 export is [84, 8400] -> transpose
        if pred.shape[0] < pred.shape[1] and pred.shape[0] in (6, 7, 8, 84, 85, 116):
            pred = pred.T

        num_features = pred.shape[1]
        if num_features < 6:
            raise RuntimeError(f"Unexpected YOLO output feature size: {pred.shape}")

        boxes = []
        scores = []
        class_ids = []

        for row in pred:
            cx, cy, w, h = row[:4]
            class_scores = row[4:]

            cls_id = int(np.argmax(class_scores))
            score = float(class_scores[cls_id])

            # If model outputs logits instead of probabilities, sigmoid can help.
            # Many YOLOv8 TFLite exports already output probabilities.
            if score > 1.0 or score < 0.0:
                class_scores = sigmoid(class_scores)
                cls_id = int(np.argmax(class_scores))
                score = float(class_scores[cls_id])

            if score < self.conf_thres:
                continue

            box = xywh_to_xyxy(cx, cy, w, h)
            boxes.append(box)
            scores.append(score)
            class_ids.append(cls_id)

        if not boxes:
            return []

        keep = nms(boxes, scores, self.iou_thres)
        boxes = [boxes[i] for i in keep]
        scores = [scores[i] for i in keep]
        class_ids = [class_ids[i] for i in keep]

        boxes = scale_boxes_to_original(boxes, ratio, dwdh, orig_shape)

        detections = []
        for box, score, cls_id in zip(boxes, scores, class_ids):
            detections.append({
                "box": box,
                "score": score,
                "class_id": cls_id,
            })

        return detections


# -----------------------------
# Drawing
# -----------------------------
def draw_detections(frame, detections, labels=None, fps=None):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        score = det["score"]
        cls_id = det["class_id"]

        if labels and 0 <= cls_id < len(labels):
            name = labels[cls_id]
        else:
            name = f"class_{cls_id}"

        text = f"{name} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(0, y1 - th - 6)
        cv2.rectangle(frame, (x1, y_text), (x1 + tw + 6, y_text + th + 6), (0, 255, 0), -1)
        cv2.putText(
            frame,
            text,
            (x1 + 3, y_text + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    if fps is not None:
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(
            frame,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


# -----------------------------
# Camera / inference loop
# -----------------------------
def camera_worker(args):
    global latest_jpeg

    labels = load_labels(args.labels)
    model = YOLOv8TFLite(
        model_path=args.model,
        conf_thres=args.conf,
        iou_thres=args.iou,
        use_edgetpu=args.edgetpu,
    )

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        detections = model.infer(frame)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        annotated = draw_detections(frame, detections, labels=labels, fps=fps)

        ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            with jpeg_lock:
                latest_jpeg = jpeg.tobytes()


# -----------------------------
# Flask routes
# -----------------------------
@app.route("/")
def index():
    html = """
    <html>
    <head>
        <title>YOLOv8 TFLite Stream</title>
        <style>
            body { font-family: Arial, sans-serif; background: #111; color: #eee; text-align: center; }
            img { max-width: 95vw; max-height: 85vh; border: 2px solid #444; margin-top: 20px; }
            .box { margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="box">
            <h1>YOLOv8 TFLite Live Stream</h1>
            <p>Browser stream from USB camera</p>
            <img src="/video_feed" />
        </div>
    </body>
    </html>
    """
    return html


def mjpeg_generator():
    global latest_jpeg
    while True:
        with jpeg_lock:
            frame = latest_jpeg

        if frame is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.01)


@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# -----------------------------
# Main
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8n TFLite browser stream")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path to YOLOv8n .tflite model")
    parser.add_argument("--labels", type=str, default=DEFAULT_LABELS, help="Optional labels txt file")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA, help="USB camera index")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Host to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind")
    parser.add_argument("--width", type=int, default=640, help="Camera capture width")
    parser.add_argument("--height", type=int, default=480, help="Camera capture height")
    parser.add_argument("--fps", type=int, default=30, help="Requested camera FPS")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS IoU threshold")
    parser.add_argument("--edgetpu", action="store_true", help="Use EdgeTPU delegate (compiled EdgeTPU model required)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    t = threading.Thread(target=camera_worker, args=(args,), daemon=True)
    t.start()

    wlan0_ip = get_ip_for_iface("wlan0")
    eth0_ip = get_ip_for_iface("eth0")

    print("\nOpen the stream in your browser:")
    print(f"  Local:   http://127.0.0.1:{args.port}")
    if wlan0_ip:
        print(f"  wlan0:   http://{wlan0_ip}:{args.port}")
    if eth0_ip:
        print(f"  eth0:    http://{eth0_ip}:{args.port}")
    print()

    app.run(host=args.host, port=args.port, threaded=True)
