import os
import time
import socket
import threading

import cv2
import numpy as np
from flask import Flask, Response

from pycoral.adapters import common
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

MODEL_PATH = "package_watcher_bv1_full_integer_quant_edgetpu.tflite"
LABELS_PATH = "labels.txt"   # set to None if you do not have labels
CAMERA_INDEX = 1
THRESHOLD = 0.3
TOP_K = 5
PORT = 5000
MAX_FPS = 15
JPEG_QUALITY = 80


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def load_labels(path):
    if path and os.path.exists(path):
        return read_label_file(path)
    return {0: "object"}


def detect_model_type(interpreter):
    output_details = interpreter.get_output_details()
    if len(output_details) >= 4:
        return "ssd_postprocess"

    shape = output_details[0]["shape"]
    if len(shape) == 3:
        return "yolo_raw"

    return "unknown"


def prepare_input(frame_bgr, interpreter):
    in_w, in_h = input_size(interpreter)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (in_w, in_h))

    input_detail = interpreter.get_input_details()[0]
    dtype = input_detail["dtype"]
    scale, zero_point = input_detail.get("quantization", (0.0, 0))

    if dtype == np.float32:
        tensor = resized.astype(np.float32) / 255.0
    elif dtype == np.uint8:
        tensor = resized.astype(np.uint8)
    elif dtype == np.int8:
        if scale and scale > 0:
            normalized = resized.astype(np.float32) / 255.0
            tensor = np.round(normalized / scale + zero_point).astype(np.int8)
        else:
            tensor = (resized.astype(np.int16) - 128).clip(-128, 127).astype(np.int8)
    else:
        tensor = resized.astype(dtype)

    common.set_input(interpreter, tensor)


def dequantize_output(array, detail):
    scale, zero_point = detail.get("quantization", (0.0, 0))
    if scale and scale > 0:
        return scale * (array.astype(np.float32) - zero_point)
    return array.astype(np.float32)


def nms_xyxy(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep


def decode_yolo_raw(interpreter, frame_shape, labels, threshold=0.25, top_k=20):
    output_detail = interpreter.get_output_details()[0]
    output = interpreter.get_tensor(output_detail["index"])
    output = dequantize_output(output, output_detail)

    if output.ndim == 3:
        output = output[0]

    # Usually Ultralytics TFLite/Edge TPU traditional output is (nc+4, N).
    if output.shape[0] < output.shape[1]:
        output = output.transpose(1, 0)

    if output.shape[1] < 5:
        return []

    boxes_xywh = output[:, :4]
    #find box mislocation bug
    

    class_scores = output[:, 4:]

    if class_scores.ndim == 1:
        class_scores = class_scores[:, None]

    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]

    keep = scores >= threshold
    boxes_xywh = boxes_xywh[keep]
    class_ids = class_ids[keep]
    scores = scores[keep]

    if len(scores) == 0:
        return []

    #bug might be here
    # print(boxes_xywh[0])

    in_w, in_h = input_size(interpreter)
    frame_h, frame_w = frame_shape[:2]
    scale_x = frame_w / float(in_w)
    scale_y = frame_h / float(in_h)

    x_center = boxes_xywh[:, 0]
    y_center = boxes_xywh[:, 1]
    width = boxes_xywh[:, 2]
    height = boxes_xywh[:, 3]

    x1 = (x_center - width / 2.0) * scale_x
    y1 = (y_center - height / 2.0) * scale_y
    x2 = (x_center + width / 2.0) * scale_x
    y2 = (y_center + height / 2.0) * scale_y


    boxes = np.stack([x1, y1, x2, y2], axis=1)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame_h - 1)

    selected = nms_xyxy(boxes, scores, iou_threshold=0.45)[:top_k]
    
    detections = []
    for i in selected:
        cls_id = int(class_ids[i])
        detections.append({
            "bbox": boxes[i],
            "score": float(scores[i]),
            "id": cls_id,
            "label": labels.get(cls_id, str(cls_id)),
        })
    print(detections["bbox"].boxes[0])
    return detections


def decode_ssd_postprocess(interpreter, frame_shape, labels, threshold=0.25, top_k=20):
    objs = get_objects(interpreter, threshold)[:top_k]
    in_w, in_h = input_size(interpreter)
    frame_h, frame_w = frame_shape[:2]
    scale_x = frame_w / float(in_w)
    scale_y = frame_h / float(in_h)

    detections = []
    for obj in objs:
        bbox = obj.bbox
        x1 = int(max(0, min(frame_w - 1, bbox.xmin * scale_x)))
        y1 = int(max(0, min(frame_h - 1, bbox.ymin * scale_y)))
        x2 = int(max(0, min(frame_w - 1, bbox.xmax * scale_x)))
        y2 = int(max(0, min(frame_h - 1, bbox.ymax * scale_y)))

        detections.append({
            "bbox": np.array([x1, y1, x2, y2], dtype=np.float32),
            "score": float(obj.score),
            "id": int(obj.id),
            "label": labels.get(int(obj.id), str(obj.id)),
        })
    return detections


def draw_detections(frame, detections, fps=None):
    h, w = frame.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        print(x1,y1,x2,y2)
        x1 = int(x1 * w)
        y1 = int(y1 * h)
        x2 = int(x2 * w)
        y2 = int(y2 * h)
        print(x1,y1,x2,y2)
        label = f'{det["label"]} {det["score"]:.2f}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.rectangle(frame, (10, 10), (20, 20), (0, 255, 0), 2)

        text_y = y1 - 10 if y1 > 20 else y1 + 25
        cv2.putText(
            frame,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "v4",
            (10, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if fps is not None:
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()

labels = load_labels(LABELS_PATH)
model_type = detect_model_type(interpreter)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

app = Flask(__name__)

latest_jpeg = None
latest_text = "Starting..."
frame_lock = threading.Lock()


def camera_worker():
    global latest_jpeg, latest_text

    frame_interval = 1.0 / MAX_FPS if MAX_FPS > 0 else 0.0
    last_fps_time = time.time()
    fps_counter = 0
    fps_value = 0.0

    while True:
        loop_start = time.time()

        ok, frame = cap.read()
        if not ok:
            latest_text = "Camera read failed"
            time.sleep(0.05)
            continue

        prepare_input(frame, interpreter)
        interpreter.invoke()

        if model_type == "ssd_postprocess":
            detections = decode_ssd_postprocess(
                interpreter=interpreter,
                frame_shape=frame.shape,
                labels=labels,
                threshold=THRESHOLD,
                top_k=TOP_K,
            )
        else:
            detections = decode_yolo_raw(
                interpreter=interpreter,
                frame_shape=frame.shape,
                labels=labels,
                threshold=THRESHOLD,
                top_k=TOP_K,
            )

        if detections:
            latest_text = ", ".join(f'{d["label"]} {d["score"]:.2f}' for d in detections)
            # print("Detected:", latest_text, detections)
        else:
            latest_text = "No objects detected"
            print(latest_text)

        fps_counter += 1
        now = time.time()
        if now - last_fps_time >= 1.0:
            fps_value = fps_counter / (now - last_fps_time)
            fps_counter = 0
            last_fps_time = now

        annotated = draw_detections(frame.copy(), detections, fps=fps_value)
        ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            with frame_lock:
                latest_jpeg = buffer.tobytes()

        elapsed = time.time() - loop_start
        if frame_interval > elapsed:
            time.sleep(frame_interval - elapsed)


def mjpeg_generator():
    while True:
        with frame_lock:
            frame = latest_jpeg

        if frame is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

        time.sleep(0.01)


@app.route("/")
def index():
    return f"""
    <html>
      <head>
        <title>Coral Edge TPU Stream</title>
        <style>
          body {{
            font-family: Arial, sans-serif;
            background: #111;
            color: #eee;
            text-align: center;
            margin: 0;
            padding: 20px;
          }}
          img {{
            width: min(95vw, 800px);
            border: 2px solid #444;
            border-radius: 8px;
          }}
          .meta {{
            color: #aaa;
            margin-bottom: 16px;
          }}
        </style>
      </head>
      <body>
        <h1>Coral Edge TPU Live Stream</h1>
        <div class="meta">Model: {MODEL_PATH}</div>
        <div class="meta">Decoder: {model_type}</div>
        <div class="meta">model input size: {input_size(interpreter)}</div>
        <div class="meta">camera input width: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
}</div>
        <div class="meta">camera input height: {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
}</div>
        <img src="/video_feed" />
      </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    threading.Thread(target=camera_worker, daemon=True).start()

    ip = get_local_ip()
    print(f"Model type: {model_type}")
    print("Open the stream in your browser:")
    print(f"  Local:   http://127.0.0.1:{PORT}")
    print(f"  Network: http://{ip}:{PORT}")

    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)