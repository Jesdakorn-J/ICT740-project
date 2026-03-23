import cv2
import tflite_runtime.interpreter as tflite
import numpy as np

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 320

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

DETECTION_MODEL_PATH = 'yolov8n_int8.tflite'

colors = [
    (47, 255, 173), (255, 255, 0), (71, 99, 255), (211, 85, 186), (180, 105, 255),
    (0, 215, 255), (87, 139, 46), (255, 105, 65), (45, 82, 160), (205, 250, 255)
]

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def get_output_tensor(interpreter, index):
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor

def detect_soms(interpreter, image, thd=0.9):
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        if scores[i] >= thd:
            results.append({
                'box': boxes[i],
                'score': scores[i]
            })
        else:
            break

    return results

def main():
    modelDetect = tflite.Interpreter(DETECTION_MODEL_PATH)
    modelDetect.allocate_tensors()

    _, input_height_detect, input_width_detect, _ = modelDetect.get_input_details()[0]['shape']

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera")
            break

        frameshow = frame.copy()

        # detection part
        imgrz = cv2.resize(
            frame.copy()[:, :, [2, 1, 0]],
            (input_width_detect, input_height_detect)
        )

        bboxes = detect_soms(modelDetect, imgrz)

        for i in range(len(bboxes)):
            bb = bboxes[i]['box']
            score = bboxes[i]['score']

            ymin, xmin, ymax, xmax = bb
            xmin = int(xmin * CAMERA_WIDTH)
            ymin = int(ymin * CAMERA_HEIGHT)
            xmax = int(xmax * CAMERA_WIDTH)
            ymax = int(ymax * CAMERA_HEIGHT)

            frameshow = cv2.rectangle(
                frameshow, (xmin, ymin), (xmax, ymax), colors[i % 10], 2
            )

            frameshow = cv2.putText(
                frameshow,
                f"detected: {score:.2f}",
                (xmin, max(ymin - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                colors[i % 10],
                2
            )

        cv2.imshow('somsom', frameshow)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
