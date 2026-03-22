#!/usr/bin/env python3
import cv2
import time


def main():
    camera_index = 0          # usually /dev/video0
    width = 640
    height = 480

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: could not open USB camera.")
        print("Check:")
        print("  1. The camera is plugged in")
        print("  2. It appears as /dev/video0")
        print("  3. No other program is using the camera")
        return

    # Try to set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    prev_time = time.time()
    fps = 0.0

    print("Camera opened successfully.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to read frame from camera.")
            break

        # Flip horizontally for a mirror-like view (optional)
        frame = cv2.flip(frame, 1)

        # FPS calculation
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time
        if dt > 0:
            fps = 1.0 / dt

        # Draw overlay text like a stream demo
        cv2.putText(
            frame,
            f"USB Camera Test | FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Coral USB Camera Stream", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
