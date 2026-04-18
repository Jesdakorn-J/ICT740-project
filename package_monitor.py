from typing import Any, Dict, Iterable, Optional

from package_tracker import PackageTracker
from telegram_sender import TelegramSender


class PackageMonitor:
    """
    Thin coordinator that connects PackageTracker and TelegramSender.

    Responsibilities:
    - accept frame + detections from your camera loop
    - ask tracker whether an event occurred
    - tell sender to notify Telegram when needed
    """

    def __init__(self, tracker: PackageTracker, sender: Optional[TelegramSender] = None) -> None:
        self.tracker = tracker
        self.sender = sender

    def process_frame(
        self,
        frame,
        detections: Optional[Iterable[Dict[str, Any]]] = None,
        motion_in_zone: bool = False,
        currently_detected: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point for your camera loop.

        Supported usage:
            monitor.process_frame(frame, detections=detections)
            monitor.process_frame(frame, detections=detections, motion_in_zone=True)
            monitor.process_frame(frame, currently_detected=True)  # legacy mode
            monitor.process_frame(frame, detections=True)          # legacy shorthand
        """
        if isinstance(detections, bool) and currently_detected is None:
            currently_detected = detections
            detections = None

        if currently_detected is not None:
            result = self.tracker.update_from_presence(
                package_seen=bool(currently_detected),
                best_box=None,
                best_score=1.0 if currently_detected else 0.0,
                motion_in_zone=motion_in_zone,
            )
        else:
            result = self.tracker.update_from_detections(
                detections=detections,
                frame_shape=frame.shape,
                motion_in_zone=motion_in_zone,
            )

        event = result.get("event")
        if event and self.sender is not None:
            sent_ok = self.sender.send_event(frame, event, metadata=result)
            result["notification_sent"] = bool(sent_ok)
        else:
            result["notification_sent"] = False

        return result


# --- TESTING BLOCK ---
if __name__ == "__main__":
    import cv2
    import numpy as np

    print("Testing PackageMonitor...")

    tracker = PackageTracker(
        package_label="package",
        detection_threshold=0.25,
        delivery_zone=(0.25, 0.25, 0.75, 0.75),
        present_threshold=8,
        empty_threshold=2,
    )
    sender = TelegramSender(dry_run=True)
    monitor = PackageMonitor(tracker=tracker, sender=sender)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "PackageMonitor Test", (105, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    sample_detection = [
        {
            "label": "package",
            "score": 0.93,
            "bbox": [240, 160, 420, 320],
        }
    ]

    print("\nSimulating package appearing:")
    for i in range(5):
        result = monitor.process_frame(dummy_frame, detections=sample_detection)
        print(f"Frame {i + 1}: {result}")

    print("\nSimulating package disappearing:")
    for i in range(10):
        result = monitor.process_frame(dummy_frame, detections=[])
        print(f"Frame {i + 1}: {result}")

    print("\nLegacy mode test:")
    tracker.reset()
    result = monitor.process_frame(dummy_frame, currently_detected=True)
    print(result)
