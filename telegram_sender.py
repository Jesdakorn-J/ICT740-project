import os
import queue
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import requests
from dotenv import load_dotenv


class TelegramSender:
    """
    Sends event notifications to Telegram without blocking the camera loop.

    Responsibilities:
    - load Telegram config
    - queue event notifications quickly from the main thread
    - send the image and caption to Telegram in a background worker thread

    It does NOT inspect detections or decide when an event happened.
    """

    EVENT_MESSAGES = {
        "APPEARED": "📦 Alert: A new package has been detected in the drop zone!",
        "DISAPPEARED": "🚨 Alert: A tracked package has been removed from the drop zone!",
    }

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_ids: Optional[Iterable[str]] = None,
        dry_run: bool = False,
        request_timeout: int = 20,
        load_env: bool = True,
        queue_size: int = 8,
        jpeg_quality: int = 90,
        worker_name: str = "telegram-sender-worker",
    ) -> None:
        if load_env:
            load_dotenv()

        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_ids = self._normalize_chat_ids(chat_ids or os.getenv("TELEGRAM_CHAT_IDS", ""))
        self.dry_run = dry_run
        self.request_timeout = request_timeout
        self.jpeg_quality = int(max(1, min(100, jpeg_quality)))
        self._queue: "queue.Queue[Optional[Tuple[bytes, str, str]]]" = queue.Queue(maxsize=max(1, queue_size))
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, name=worker_name, daemon=True)
        self._worker.start()

    def _normalize_chat_ids(self, chat_ids: Any) -> List[str]:
        if chat_ids is None:
            return []

        if isinstance(chat_ids, str):
            return [item.strip() for item in chat_ids.split(",") if item.strip()]

        normalized = []
        for item in chat_ids:
            value = str(item).strip()
            if value:
                normalized.append(value)
        return normalized

    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_ids)

    def build_caption(self, event_type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        message = self.EVENT_MESSAGES.get(event_type, "Camera Alert: Activity detected.")

        if not metadata:
            return message

        score = metadata.get("package_score")
        best_score = metadata.get("best_score")

        extras = []
        if score is not None:
            extras.append(f"tracker_score={score}")
        if best_score is not None:
            extras.append(f"det_conf={float(best_score):.2f}")

        if extras:
            message = f"{message}\n\n" + " | ".join(extras)
        return message

    def _encode_frame_to_jpeg_bytes(self, frame) -> Optional[bytes]:
        if frame is None:
            return None

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return None
        return encoded.tobytes()

    def send_event(
        self,
        frame,
        event_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Queue one event notification and return immediately.

        Returns True if the task was accepted into the queue (or if dry_run=True),
        else False. The actual network send happens in the background worker.
        """
        caption = self.build_caption(event_type, metadata)

        if self.dry_run:
            print(f"[DRY RUN] Telegram event queued: {event_type}")
            print(f"[DRY RUN] Caption: {caption}")
            if frame is not None:
                jpg_bytes = self._encode_frame_to_jpeg_bytes(frame)
                if jpg_bytes is not None:
                    print(f"[DRY RUN] Encoded JPEG bytes: {len(jpg_bytes)} bytes")
                else:
                    print("[DRY RUN] Failed to encode frame to JPEG")
            return True

        if not self.is_configured():
            print("Error: Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_IDS.")
            return False

        jpg_bytes = self._encode_frame_to_jpeg_bytes(frame)
        if jpg_bytes is None:
            print("Error: frame is required and must be encodable to send a Telegram photo alert.")
            return False

        try:
            self._queue.put_nowait((jpg_bytes, caption, event_type))
            return True
        except queue.Full:
            print("Warning: Telegram queue is full. Dropping alert to protect FPS.")
            return False

    def _worker_loop(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    break

                jpg_bytes, caption, event_type = task
                self._send_encoded_photo(jpg_bytes, caption, event_type)
            except Exception as exc:
                print(f"Worker error while sending Telegram alert: {exc}")
            finally:
                self._queue.task_done()

    def _send_encoded_photo(self, jpg_bytes: bytes, caption: str, event_type: str) -> bool:
        url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        overall_success = True

        for chat_id in self.chat_ids:
            payload = {
                "chat_id": chat_id,
                "caption": caption,
            }
            files = {
                "photo": ("snapshot.jpg", jpg_bytes, "image/jpeg")
            }

            try:
                response = requests.post(
                    url,
                    data=payload,
                    files=files,
                    timeout=self.request_timeout,
                )

                if response.status_code == 200:
                    print(f"Success! Sent {event_type} alert to chat {chat_id}.")
                else:
                    overall_success = False
                    print(f"Failed to send to {chat_id}. Status code: {response.status_code}")
                    print(response.text)
            except Exception as exc:
                overall_success = False
                print(f"Error sending Telegram alert to {chat_id}: {exc}")

        return overall_success

    def flush(self) -> None:
        """Block until all queued Telegram sends are finished."""
        self._queue.join()

    def shutdown(self, wait: bool = True) -> None:
        """
        Stop the background worker.

        wait=True will first finish queued tasks, then stop the worker.
        """
        if wait:
            self.flush()

        if self._worker.is_alive() and not self._stop_event.is_set():
            self._stop_event.set()
            self._queue.put(None)
            self._worker.join(timeout=self.request_timeout + 5)

    def __del__(self):
        try:
            self.shutdown(wait=False)
        except Exception:
            pass


# --- TESTING BLOCK ---
if __name__ == "__main__":
    import time

    import numpy as np

    print("Testing TelegramSender with background thread...")

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "TelegramSender Thread Test", (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    sender = TelegramSender(dry_run=True, queue_size=4)

    for i in range(3):
        ok = sender.send_event(
            dummy_frame,
            "APPEARED" if i < 2 else "DISAPPEARED",
            metadata={"package_score": 8 - i, "best_score": 0.91 - (i * 0.1)},
        )
        print(f"Queue result #{i + 1}: {ok}")

    # In dry-run mode nothing is actually queued to the worker, but keep this here
    # to show the normal shutdown pattern for real use.
    time.sleep(0.2)
    sender.shutdown(wait=True)
    print("TelegramSender test finished.")
