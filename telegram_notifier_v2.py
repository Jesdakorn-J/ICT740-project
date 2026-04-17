import requests
import cv2
import os
import numpy as np
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()


class TelegramNotifier:
    def __init__(
        self,
        frames_to_confirm=5,
        package_label="package",
        detection_threshold=0.25,
        delivery_zone=None,
        score_add=2,
        score_subtract=1,
        score_max=20,
        present_threshold=8,
        empty_threshold=2,
    ):
        """
        Smart Telegram notifier using a score-based state machine.

        Args:
            frames_to_confirm: Kept for backward compatibility with older code.
            package_label: Class name that should be treated as the package.
            detection_threshold: Minimum confidence to count as a package.
            delivery_zone:
                Optional delivery zone rectangle.
                - None: whole frame is valid.
                - (x1, y1, x2, y2) in pixels, or
                - normalized floats from 0.0 to 1.0.
            score_add: Score increase when a valid package is seen.
            score_subtract: Score decrease when no valid package is seen.
            score_max: Maximum package score.
            present_threshold: Score needed to consider a package present.
            empty_threshold: Score low enough to consider the area empty again.
        """
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_ids_string = os.getenv("TELEGRAM_CHAT_IDS", "")
        self.chat_ids = [chat_id.strip() for chat_id in chat_ids_string.split(",") if chat_id.strip()]

        # Backward-compatible field from the previous version.
        self.frames_to_confirm = frames_to_confirm

        # Detection config.
        self.package_label = package_label
        self.detection_threshold = detection_threshold
        self.delivery_zone = delivery_zone

        # Score system config.
        self.score_add = score_add
        self.score_subtract = score_subtract
        self.score_max = score_max
        self.present_threshold = present_threshold
        self.empty_threshold = empty_threshold

        # State machine memory.
        self.state = "EMPTY"
        self.package_score = 0
        self.notify_sent = False

        # Debug info from the latest processed frame.
        self.last_best_box = None
        self.last_best_score = 0.0

    def set_delivery_zone(self, delivery_zone):
        """Update the delivery zone at runtime."""
        self.delivery_zone = delivery_zone

    def _resolve_delivery_zone(self, frame_shape):
        """
        Converts the configured delivery zone to pixel coordinates.
        Returns None if the whole frame should be treated as valid.
        """
        if self.delivery_zone is None:
            return None

        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2 = self.delivery_zone

        # Support normalized coordinates like (0.2, 0.3, 0.8, 0.9)
        if all(isinstance(v, (int, float)) for v in (x1, y1, x2, y2)) and all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)):
            x1 = int(x1 * frame_w)
            y1 = int(y1 * frame_h)
            x2 = int(x2 * frame_w)
            y2 = int(y2 * frame_h)

        x1 = max(0, min(int(x1), frame_w - 1))
        y1 = max(0, min(int(y1), frame_h - 1))
        x2 = max(0, min(int(x2), frame_w - 1))
        y2 = max(0, min(int(y2), frame_h - 1))
        return x1, y1, x2, y2

    def box_center_in_delivery_zone(self, x1, y1, x2, y2, frame_shape):
        """Checks whether the center of a box lies inside the delivery zone."""
        zone = self._resolve_delivery_zone(frame_shape)
        if zone is None:
            return True

        zx1, zy1, zx2, zy2 = zone
        cx = (float(x1) + float(x2)) / 2.0
        cy = (float(y1) + float(y2)) / 2.0
        return zx1 <= cx <= zx2 and zy1 <= cy <= zy2

    def _package_seen_from_detections(self, detections, frame_shape):
        """
        Returns:
            package_seen (bool), best_box, best_score
        """
        package_seen = False
        best_box = None
        best_score = 0.0

        if not detections:
            return package_seen, best_box, best_score

        for det in detections:
            label = det.get("label")
            if label != self.package_label:
                continue

            bbox = det.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            conf = float(det.get("score", 0.0))

            if conf < self.detection_threshold:
                continue

            if not self.box_center_in_delivery_zone(x1, y1, x2, y2, frame_shape):
                continue

            package_seen = True
            if conf > best_score:
                best_score = conf
                best_box = bbox

        return package_seen, best_box, best_score

    def _update_score(self, package_seen):
        if package_seen:
            self.package_score = min(self.package_score + self.score_add, self.score_max)
        else:
            self.package_score = max(self.package_score - self.score_subtract, 0)

    def process_frame(self, frame, detections=None, motion_in_zone=False, currently_detected=None):
        """
        Process one frame and update the package state.

        Supported usage:
            process_frame(frame, detections=[...])
            process_frame(frame, detections=[...], motion_in_zone=False)
            process_frame(frame, currently_detected=True)
            process_frame(frame, True)   # legacy bool shorthand

        Args:
            frame: Current image frame.
            detections: Detection list like:
                [{"label": "package", "score": 0.83, "bbox": [x1, y1, x2, y2]}, ...]
            motion_in_zone: Accepted for compatibility with your planned pipeline.
                            It is not used by this score logic yet.
            currently_detected: Legacy bool input from the older version.

        Returns:
            dict with state, score, whether package was seen, and any event fired.
        """
        if isinstance(detections, bool) and currently_detected is None:
            currently_detected = detections
            detections = None

        if currently_detected is not None:
            package_seen = bool(currently_detected)
            best_box = None
            best_score = 1.0 if package_seen else 0.0
        else:
            package_seen, best_box, best_score = self._package_seen_from_detections(detections, frame.shape)

        self.last_best_box = best_box
        self.last_best_score = best_score

        self._update_score(package_seen)

        event_fired = None
        previous_state = self.state
        previous_notify_sent = self.notify_sent

        if self.state == "EMPTY":
            if self.package_score >= self.present_threshold:
                self.state = "PRESENT"
                self.notify_sent = False

        elif self.state == "PRESENT":
            if not self.notify_sent:
                print("📦 Triggering APPEARED alert to Telegram!")
                self._send_alert(frame, "APPEARED")
                self.notify_sent = True
                self.state = "NOTIFIED"
                event_fired = "APPEARED"

            if self.package_score <= self.empty_threshold:
                self.state = "EMPTY"

        elif self.state == "NOTIFIED":
            if self.package_score <= self.empty_threshold:
                self.state = "EMPTY"
                self.notify_sent = False

        # Keep the removal alert behavior from your older notifier.
        if self.state == "EMPTY" and previous_state == "NOTIFIED" and previous_notify_sent:
            print("🚨 Triggering DISAPPEARED alert to Telegram!")
            self._send_alert(frame, "DISAPPEARED")
            event_fired = "DISAPPEARED"

        debug = {
            "state": self.state,
            "package_score": self.package_score,
            "notify_sent": self.notify_sent,
            "package_seen": package_seen,
            "best_box": best_box,
            "best_score": best_score,
            "motion_in_zone": motion_in_zone,
            "event": event_fired,
        }
        return debug

    def _send_alert(self, frame, event_type):
        """
        The actual HTTP request to send the image to Telegram.
        (This is kept internal to the class)
        """
        if not self.bot_token or not self.chat_ids:
            print("Error: Missing Bot Token or Chat IDs in .env file.")
            return

        temp_image_path = "temp_snapshot.jpg"
        cv2.imwrite(temp_image_path, frame)
        url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"

        if event_type == "APPEARED":
            message = "📦 Alert: A new package has been detected in the drop zone!"
        elif event_type == "DISAPPEARED":
            message = "🚨 Alert: A tracked package has been removed from the drop zone!"
        else:
            message = "Camera Alert: Activity detected."

        try:
            for chat_id in self.chat_ids:
                payload = {"chat_id": chat_id, "caption": message}

                with open(temp_image_path, "rb") as photo:
                    files = {"photo": photo}
                    response = requests.post(url, data=payload, files=files)

                if response.status_code == 200:
                    print(f"Success! Sent {event_type} alert to User {chat_id}.")
                else:
                    print(f"Failed to send to {chat_id}. Status code: {response.status_code}")
                    print(response.text)

        except Exception as e:
            print(f"Error sending Telegram alert: {e}")

        finally:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)


# --- TESTING BLOCK ---
if __name__ == "__main__":
    print("Testing score-based Telegram Notifier...")

    notifier = TelegramNotifier(
        detection_threshold=0.25,
        delivery_zone=(0.25, 0.25, 0.75, 0.75),  # normalized zone example
        present_threshold=8,
        empty_threshold=2,
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "Test Snapshot", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    sample_detection = [{
        "label": "package",
        "score": 0.90,
        "bbox": [240, 160, 420, 320],
    }]

    print("Simulating package appearing...")
    for i in range(5):
        result = notifier.process_frame(dummy_frame, detections=sample_detection)
        print(f"Frame {i + 1}: {result}")

    print("Simulating package disappearing...")
    for i in range(10):
        result = notifier.process_frame(dummy_frame, detections=[])
        print(f"Frame {i + 1}: {result}")
