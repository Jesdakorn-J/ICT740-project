from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class PackageTracker:
    """
    Tracks whether a package is present using a score-based state machine.

    Responsibilities:
    - filter detections
    - apply delivery-zone rules
    - update package score
    - emit logical events: APPEARED / DISAPPEARED

    It does NOT send Telegram messages or perform any network I/O.
    """

    def __init__(
        self,
        package_label: str = "package",
        detection_threshold: float = 0.25,
        delivery_zone: Optional[Sequence[float]] = None,
        score_add: int = 2,
        score_subtract: int = 1,
        score_max: int = 20,
        present_threshold: int = 8,
        empty_threshold: int = 2,
        require_motion_in_zone: bool = False,
    ) -> None:
        self.package_label = package_label
        self.detection_threshold = detection_threshold
        self.delivery_zone = delivery_zone
        self.score_add = score_add
        self.score_subtract = score_subtract
        self.score_max = score_max
        self.present_threshold = present_threshold
        self.empty_threshold = empty_threshold
        self.require_motion_in_zone = require_motion_in_zone

        self.state = "EMPTY"
        self.package_score = 0
        self.notify_sent = False

        self.last_best_box = None
        self.last_best_score = 0.0
        self.last_package_seen = False

    def reset(self) -> None:
        """Reset the tracker to its initial state."""
        self.state = "EMPTY"
        self.package_score = 0
        self.notify_sent = False
        self.last_best_box = None
        self.last_best_score = 0.0
        self.last_package_seen = False

    def set_delivery_zone(self, delivery_zone: Optional[Sequence[float]]) -> None:
        """Update the delivery zone at runtime."""
        self.delivery_zone = delivery_zone

    def _resolve_delivery_zone(self, frame_shape: Sequence[int]) -> Optional[Tuple[int, int, int, int]]:
        """
        Convert delivery_zone into pixel coordinates.

        Supported formats:
        - None -> whole frame is valid
        - (x1, y1, x2, y2) in pixels
        - (x1, y1, x2, y2) normalized between 0.0 and 1.0
        """
        if self.delivery_zone is None:
            return None

        if len(self.delivery_zone) != 4:
            raise ValueError("delivery_zone must be None or a 4-value sequence")

        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2 = self.delivery_zone

        is_normalized = all(isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0 for v in (x1, y1, x2, y2))
        if is_normalized:
            x1 = int(float(x1) * frame_w)
            y1 = int(float(y1) * frame_h)
            x2 = int(float(x2) * frame_w)
            y2 = int(float(y2) * frame_h)
        else:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        x1 = max(0, min(x1, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        x2 = max(0, min(x2, frame_w - 1))
        y2 = max(0, min(y2, frame_h - 1))

        if x2 < x1 or y2 < y1:
            raise ValueError("delivery_zone must satisfy x1 <= x2 and y1 <= y2")

        return x1, y1, x2, y2

    def box_center_in_delivery_zone(self, bbox: Sequence[float], frame_shape: Sequence[int]) -> bool:
        """Return True if the box center is inside the delivery zone."""
        zone = self._resolve_delivery_zone(frame_shape)
        if zone is None:
            return True

        x1, y1, x2, y2 = bbox
        zx1, zy1, zx2, zy2 = zone
        cx = (float(x1) + float(x2)) / 2.0
        cy = (float(y1) + float(y2)) / 2.0
        return zx1 <= cx <= zx2 and zy1 <= cy <= zy2

    def find_best_package(
        self,
        detections: Optional[Iterable[Dict[str, Any]]],
        frame_shape: Sequence[int],
    ) -> Tuple[bool, Optional[Sequence[float]], float]:
        """
        Inspect detections and return:
            package_seen, best_box, best_score

        Expected detection format:
            {
                "label": "package",
                "score": 0.92,
                "bbox": [x1, y1, x2, y2],
            }
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

            conf = float(det.get("score", 0.0))
            if conf < self.detection_threshold:
                continue

            if not self.box_center_in_delivery_zone(bbox, frame_shape):
                continue

            package_seen = True
            if conf > best_score:
                best_score = conf
                best_box = bbox

        return package_seen, best_box, best_score

    def _update_score(self, package_seen: bool, motion_in_zone: bool = False) -> None:
        """
        Update score memory.

        If require_motion_in_zone is enabled, positive scoring only happens when both
        package_seen and motion_in_zone are True.
        """
        positive_hit = package_seen
        if self.require_motion_in_zone:
            positive_hit = package_seen and motion_in_zone

        if positive_hit:
            self.package_score = min(self.package_score + self.score_add, self.score_max)
        else:
            self.package_score = max(self.package_score - self.score_subtract, 0)

    def update_from_presence(
        self,
        package_seen: bool,
        best_box: Optional[Sequence[float]] = None,
        best_score: float = 0.0,
        motion_in_zone: bool = False,
    ) -> Dict[str, Any]:
        """
        Update the state machine from an already-decided package_seen flag.

        Returns a dict containing the new state and any emitted event.
        """
        self.last_package_seen = bool(package_seen)
        self.last_best_box = best_box
        self.last_best_score = float(best_score)

        previous_state = self.state
        self._update_score(package_seen=package_seen, motion_in_zone=motion_in_zone)

        event = None

        if self.state == "EMPTY" and self.package_score >= self.present_threshold:
            self.state = "PRESENT"
            self.notify_sent = False

        if self.state == "PRESENT" and not self.notify_sent:
            event = "APPEARED"
            self.notify_sent = True
            self.state = "NOTIFIED"

        if self.state == "NOTIFIED" and self.package_score <= self.empty_threshold:
            event = "DISAPPEARED"
            self.state = "EMPTY"
            self.notify_sent = False

        if self.state == "PRESENT" and self.package_score <= self.empty_threshold:
            # Safety guard in case thresholds are configured oddly.
            self.state = "EMPTY"
            self.notify_sent = False

        return {
            "event": event,
            "state": self.state,
            "previous_state": previous_state,
            "package_score": self.package_score,
            "package_seen": bool(package_seen),
            "best_box": best_box,
            "best_score": float(best_score),
            "notify_sent": self.notify_sent,
            "motion_in_zone": bool(motion_in_zone),
        }

    def update_from_detections(
        self,
        detections: Optional[Iterable[Dict[str, Any]]],
        frame_shape: Sequence[int],
        motion_in_zone: bool = False,
    ) -> Dict[str, Any]:
        """Find the best package from detections, then update the state machine."""
        package_seen, best_box, best_score = self.find_best_package(detections, frame_shape)
        return self.update_from_presence(
            package_seen=package_seen,
            best_box=best_box,
            best_score=best_score,
            motion_in_zone=motion_in_zone,
        )


# --- TESTING BLOCK ---
if __name__ == "__main__":
    print("Testing PackageTracker...")

    tracker = PackageTracker(
        package_label="package",
        detection_threshold=0.25,
        delivery_zone=(0.25, 0.25, 0.75, 0.75),
        present_threshold=8,
        empty_threshold=2,
    )

    frame_shape = (480, 640, 3)
    inside_zone_detection = [
        {
            "label": "package",
            "score": 0.91,
            "bbox": [240, 160, 420, 320],
        }
    ]
    outside_zone_detection = [
        {
            "label": "package",
            "score": 0.99,
            "bbox": [10, 10, 60, 60],
        }
    ]

    print("\nSimulating package inside the zone:")
    for i in range(5):
        result = tracker.update_from_detections(inside_zone_detection, frame_shape)
        print(f"Frame {i + 1}: {result}")

    print("\nSimulating false package outside the zone:")
    result = tracker.update_from_detections(outside_zone_detection, frame_shape)
    print(result)

    print("\nSimulating package disappearing:")
    for i in range(10):
        result = tracker.update_from_detections([], frame_shape)
        print(f"Frame {i + 1}: {result}")
