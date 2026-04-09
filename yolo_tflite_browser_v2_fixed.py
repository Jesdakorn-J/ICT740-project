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

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ("network", ip) not in results and ip != "127.0.0.1":
            results.append(("network", ip))
    except Exception:
        pass

    seen = set()
    uniq = []
    for name, ip in results:
        if ip not in seen:
            uniq.append((name, ip))
            seen.add(ip)
    return uniq


def letterbox(image: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
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
    def __init__(self, model_path: str, conf_thres=0.25, iou_thres=0.45, labels=None, debug=False):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.labels = labels or COCO80
        self.debug = debug
        self._logged_shapes = False

        delegates = []
        try:
            delegates = [load_delegate("libedgetpu.so.1")]
            if self.debug:
                print("Loaded EdgeTPU delegate")
        except Exception as e:
            print(f"Warning: could not load EdgeTPU delegate: {e}")
            print("Falling back to CPU TFLite interpreter")

        self.interpreter = Interpreter(model_path=model_path, experimental_delegates=delegates)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_index = self.input_details[0]["index"]

        input_shape = self.input_details[0]["shape"]
        self.in_h = int(input_shape[1])
        self.in_w = int(input_shape[2])

        self.input_dtype = self.input_details[0]["dtype"]
        self.input_quant = self.input_details[0]["quantization"]

        if self.debug:
            print("Input details:")
            for d in self.input_details:
                print(f"  shape={d['shape']} dtype={d['dtype']} quant={d['quantization']}")
            print("Output details:")
            for i, d in enumerate(self.output_details):
                print(f"  [{i}] shape={d['shape']} dtype={d['dtype']} quant={d['quantization']}")

    def preprocess(self, frame: np.ndarray):
        img, ratio, pad_x, pad_y = letterbox(frame, (self.in_h, self.in_w))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.input_dtype == np.uint8:
            scale, zero = self.input_quant
            if scale == 0:
                inp = rgb.astype(np.uint8)
            else:
                inp = np.clip(np.round(rgb / scale + zero), 0, 255).astype(np.uint8)
        elif self.input_dtype == np.int8:
            scale, zero = self.input_quant
            if scale == 0:
                inp = rgb.astype(np.int8)
            else:
                inp = np.clip(np.round(rgb / scale + zero), -128, 127).astype(np.int8)
        else:
            inp = rgb.astype(np.float32) / 255.0

        inp = np.expand_dims(inp, axis=0)
        return inp, ratio, pad_x, pad_y

    def _get_output_tensor(self, detail):
        out = self.interpreter.get_tensor(detail["index"])
        dtype = detail["dtype"]
        scale, zero = detail.get("quantization", (0.0, 0))

        if dtype == np.uint8 and scale != 0:
            return (out.astype(np.float32) - zero) * scale
        if dtype == np.int8 and scale != 0:
            return (out.astype(np.float32) - zero) * scale
        return out.astype(np.float32) if dtype != np.float32 else out

    def infer(self, frame: np.ndarray):
        orig_h, orig_w = frame.shape[:2]
        inp, ratio, pad_x, pad_y = self.preprocess(frame)

        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        outputs = [self._get_output_tensor(d) for d in self.output_details]

        if self.debug and not self._logged_shapes:
            print("Runtime output shapes:", [o.shape for o in outputs])
            self._logged_shapes = True

        boxes, scores, class_ids = self.decode_output(outputs, orig_w, orig_h, ratio, pad_x, pad_y)
        return boxes, scores, class_ids

    def decode_output(self, outputs, orig_w, orig_h, ratio, pad_x, pad_y):
        if len(outputs) >= 3:
            decoded = self.try_decode_detection_api(outputs, orig_w, orig_h)
            if decoded is not None:
                return decoded

        for out in outputs:
            decoded = self.try_decode_yolo_matrix(out, orig_w, orig_h, ratio, pad_x, pad_y)
            if decoded is not None:
                return decoded

        if self.debug:
            print("Could not decode model outputs. Shapes were:", [o.shape for o in outputs])
        return np.empty((0, 4)), np.array([]), np.array([])

    def try_decode_detection_api(self, outputs, orig_w, orig_h):
        """Handles TFLite Detection_PostProcess style outputs:
        boxes [1,N,4], classes [1,N], scores [1,N], count [1]
        """
        boxes = classes = scores = count = None

        for out in outputs:
            s = tuple(out.shape)
            if len(s) == 3 and s[0] == 1 and s[-1] == 4:
                boxes = np.squeeze(out, axis=0)
            elif len(s) == 2 and s[0] == 1:
                arr = np.squeeze(out, axis=0)
                if np.issubdtype(arr.dtype, np.floating):
                    if np.all(arr >= 0) and np.all(arr <= 1.01):
                        scores = arr
                    else:
                        classes = arr.astype(np.int32)
                else:
                    classes = arr.astype(np.int32)
            elif np.size(out) == 1:
                count = int(np.squeeze(out))

        if boxes is None or classes is None or scores is None:
            return None

        n = min(len(boxes), len(classes), len(scores))
        if count is not None:
            n = min(n, count)

        boxes = boxes[:n]
        classes = classes[:n].astype(np.int32)
        scores = scores[:n]

        mask = scores >= self.conf_thres
        boxes = boxes[mask]
        classes = classes[mask]
        scores = scores[mask]

        if len(boxes) == 0:
            return np.empty((0, 4)), np.array([]), np.array([])

        # Usually normalized ymin, xmin, ymax, xmax
        if np.max(boxes) <= 1.5:
            y1 = boxes[:, 0] * orig_h
            x1 = boxes[:, 1] * orig_w
            y2 = boxes[:, 2] * orig_h
            x2 = boxes[:, 3] * orig_w
        else:
            y1 = boxes[:, 0]
            x1 = boxes[:, 1]
            y2 = boxes[:, 2]
            x2 = boxes[:, 3]

        final_boxes = np.stack([
            np.clip(x1, 0, orig_w - 1),
            np.clip(y1, 0, orig_h - 1),
            np.clip(x2, 0, orig_w - 1),
            np.clip(y2, 0, orig_h - 1),
        ], axis=1)

        keep = nms(final_boxes, scores, self.iou_thres)
        return final_boxes[keep], scores[keep], classes[keep]

    def try_decode_yolo_matrix(self, output, orig_w, orig_h, ratio, pad_x, pad_y):
        pred = np.squeeze(output)
        if pred.ndim != 2:
            return None

        # Bring to [N, C]
        if pred.shape[0] in (84, 85) and pred.shape[1] > 20:
            pred = pred.T
        elif pred.shape[1] in (84, 85):
            pass
        else:
            return None

        if pred.shape[1] == 84:
            boxes_xywh = pred[:, :4]
            class_scores = pred[:, 4:]
            class_ids = np.argmax(class_scores, axis=1)
            scores = class_scores[np.arange(len(class_scores)), class_ids]
        elif pred.shape[1] == 85:
            boxes_xywh = pred[:, :4]
            objectness = pred[:, 4]
            class_scores = pred[:, 5:]
            class_ids = np.argmax(class_scores, axis=1)
            scores = objectness * class_scores[np.arange(len(class_scores)), class_ids]
        else:
            return None

        mask = scores >= self.conf_thres
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return np.empty((0, 4)), np.array([]), np.array([])

        # Heuristic: some exports output normalized xywh, others in input-pixel space.
        if np.max(np.abs(boxes_xywh[:, :4])) <= 2.0:
            x = boxes_xywh[:, 0] * self.in_w
            y = boxes_xywh[:, 1] * self.in_h
            w = boxes_xywh[:, 2] * self.in_w
            h = boxes_xywh[:, 3] * self.in_h
        else:
            x = boxes_xywh[:, 0]
            y = boxes_xywh[:, 1]
            w = boxes_xywh[:, 2]
            h = boxes_xywh[:, 3]

        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        x1 = (x1 - pad_x) / ratio
        y1 = (y1 - pad_y) / ratio
        x2 = (x2 - pad_x) / ratio
        y2 = (y2 - pad_y) / ratio

        boxes = np.stack([
            np.clip(x1, 0, orig_w - 1),
            np.clip(y1, 0, orig_h - 1),
            np.clip(x2, 0, orig_w - 1),
            np.clip(y2, 0, orig_h - 1),
        ], axis=1)

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
        cv2.putText(frame, text, (x1, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


class StreamState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_jpeg = None
        self.fps = 0.0
        self.running = True
        self.last_count = 0


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

            yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
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
        labels=COCO80,
        debug=args.debug,
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
        state.last_count = len(boxes)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Detections: {len(boxes)}", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with state.lock:
                state.frame_jpeg = jpeg.tobytes()

    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n_full_integer_quant_edgetpu.tflite", help="Path to EdgeTPU YOLOv8 TFLite model")
    parser.add_argument("--camera", type=int, default=0, help="USB camera index")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--host", default="0.0.0.0", help="Flask bind host")
    parser.add_argument("--port", type=int, default=5000, help="Flask port")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--debug", action="store_true", help="Print input/output tensor info")
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
