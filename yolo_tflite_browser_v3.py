import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

MODEL_PATH = "yolov8n_full_integer_quant_edgetpu.tflite"
CONF_THRES = 0.4
IOU_THRES = 0.5

colors = [
    (47, 255, 173), (255, 255, 0), (71, 99, 255), (211, 85, 186), (180, 105, 255),
    (0, 215, 255), (87, 139, 46), (255, 105, 65), (45, 82, 160), (205, 250, 255)
]

# Optional: replace with your own class names if needed
CLASS_NAMES = None
# Example:
# CLASS_NAMES = ["person", "bicycle", "car", ...]

def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = image.shape[:2]
    new_w, new_h = new_shape

    r = min(new_w / w, new_h / h)
    resized_w, resized_h = int(round(w * r)), int(round(h * r))

    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    dw = new_w - resized_w
    dh = new_h - resized_h
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

    return padded, r, left, top

def compute_iou(box, boxes):
    # box: [x1,y1,x2,y2]
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = box_area + boxes_area - inter + 1e-6
    return inter / union

def non_max_suppression(boxes, scores, class_ids, iou_thres=0.5):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    class_ids = np.array(class_ids, dtype=np.int32)

    keep = []
    unique_classes = np.unique(class_ids)

    for cls in unique_classes:
        idxs = np.where(class_ids == cls)[0]
        cls_boxes = boxes[idxs]
        cls_scores = scores[idxs]

        order = np.argsort(-cls_scores)

        while len(order) > 0:
            i = order[0]
            keep.append(idxs[i])

            if len(order) == 1:
                break

            ious = compute_iou(cls_boxes[i], cls_boxes[order[1:]])
            order = order[1:][ious < iou_thres]

    return keep

def scale_boxes(boxes, ratio, pad_x, pad_y, orig_w, orig_h):
    scaled = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = (x1 - pad_x) / ratio
        y1 = (y1 - pad_y) / ratio
        x2 = (x2 - pad_x) / ratio
        y2 = (y2 - pad_y) / ratio

        x1 = max(0, min(orig_w - 1, x1))
        y1 = max(0, min(orig_h - 1, y1))
        x2 = max(0, min(orig_w - 1, x2))
        y2 = max(0, min(orig_h - 1, y2))

        scaled.append([int(x1), int(y1), int(x2), int(y2)])
    return scaled

def parse_yolov8_output(output, conf_thres=0.4):
    """
    Supports common YOLOv8 output shapes:
    - (1, 84, 8400)
    - (1, 8400, 84)
    - squeezed versions of the above
    """

    output = np.squeeze(output)

    if output.ndim != 2:
        raise ValueError(f"Unexpected output shape after squeeze: {output.shape}")

    # Convert to [num_preds, num_attrs]
    if output.shape[0] < output.shape[1]:
        # likely [84, 8400] -> transpose to [8400, 84]
        output = output.T

    # Expected attrs: [cx, cy, w, h, class_probs...]
    if output.shape[1] < 6:
        raise ValueError(f"Unexpected YOLOv8 output shape: {output.shape}")

    boxes = []
    scores = []
    class_ids = []

    for pred in output:
        cx, cy, w, h = pred[:4]
        class_scores = pred[4:]

        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id])

        if score < conf_thres:
            continue

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        class_ids.append(class_id)

    return boxes, scores, class_ids

def main():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    interpreter = Interpreter(
        model_path=MODEL_PATH,
        experimental_delegates=[load_delegate("libedgetpu.so.1")]
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    _, input_h, input_w, input_c = input_shape

    print("Input shape:", input_shape)
    print("Output details:", output_details)

    input_dtype = input_details[0]["dtype"]
    input_scale, input_zero_point = input_details[0]["quantization"]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera")
            break

        orig_h, orig_w = frame.shape[:2]
        frame_show = frame.copy()

        # BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize with padding
        img, ratio, pad_x, pad_y = letterbox(rgb, (input_w, input_h))

        # Add batch dimension
        input_data = np.expand_dims(img, axis=0)

        # Quantize input if needed
        if input_dtype == np.uint8:
            if input_scale > 0:
                input_data = input_data / input_scale + input_zero_point
            input_data = np.clip(input_data, 0, 255).astype(np.uint8)
        elif input_dtype == np.int8:
            if input_scale > 0:
                input_data = input_data / input_scale + input_zero_point
            input_data = np.clip(input_data, -128, 127).astype(np.int8)
        else:
            input_data = input_data.astype(input_dtype)

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]["index"])

        try:
            boxes, scores, class_ids = parse_yolov8_output(output, CONF_THRES)
        except Exception as e:
            cv2.putText(
                frame_show,
                f"Output parse error: {str(e)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            cv2.imshow("YOLOv8n EdgeTPU", frame_show)
            if cv2.waitKey(1) == 27:
                break
            continue

        keep = non_max_suppression(boxes, scores, class_ids, IOU_THRES)

        boxes = [boxes[i] for i in keep]
        scores = [scores[i] for i in keep]
        class_ids = [class_ids[i] for i in keep]

        boxes = scale_boxes(boxes, ratio, pad_x, pad_y, orig_w, orig_h)

        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            x1, y1, x2, y2 = box
            color = colors[i % len(colors)]

            if CLASS_NAMES and 0 <= class_id < len(CLASS_NAMES):
                label = f"{CLASS_NAMES[class_id]} {score:.2f}"
            else:
                label = f"id:{class_id} {score:.2f}"

            cv2.rectangle(frame_show, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame_show,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        cv2.imshow("YOLOv8n EdgeTPU", frame_show)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
