import requests
import cv2
import os
import numpy as np
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

class TelegramNotifier:
    def __init__(self, frames_to_confirm=5):
        """
        Initializes the notifier and sets up the tracking memory.
        frames_to_confirm: How many consecutive frames the package must be 
                           seen (or missing) before sending an alert.
        """
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_ids_string = os.getenv('TELEGRAM_CHAT_IDS', '')
        self.chat_ids = [chat_id.strip() for chat_id in chat_ids_string.split(',') if chat_id.strip()]
        
        # --- State Tracking Variables ---
        self.frames_to_confirm = frames_to_confirm
        self.package_present = False
        self.consecutive_detects = 0
        self.consecutive_misses = 0

    def process_frame(self, frame, currently_detected):
        """
        Takes the current frame and whether a package was detected.
        Updates the internal counters and decides if an alert should be sent.
        """
        if currently_detected:
            self.consecutive_misses = 0
            self.consecutive_detects += 1
            
            # If we've seen it enough times, and haven't alerted yet
            if self.consecutive_detects >= self.frames_to_confirm and not self.package_present:
                self.package_present = True
                print("📦 Triggering APPEARED alert to Telegram!")
                self._send_alert(frame, "APPEARED")
                
        else:
            self.consecutive_detects = 0
            self.consecutive_misses += 1
            
            # If it's been gone enough times, and we previously knew it was there
            if self.consecutive_misses >= self.frames_to_confirm and self.package_present:
                self.package_present = False
                print("🚨 Triggering DISAPPEARED alert to Telegram!")
                self._send_alert(frame, "DISAPPEARED")

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
                
                with open(temp_image_path, 'rb') as photo:
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
    print("Testing Smart Telegram Notifier...")
    notifier = TelegramNotifier(frames_to_confirm=2) # Shortened for testing
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8) 
    cv2.putText(dummy_frame, "Test Snapshot", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    print("Simulating package appearing...")
    notifier.process_frame(dummy_frame, currently_detected=True) # Frame 1 (No alert yet)
    notifier.process_frame(dummy_frame, currently_detected=True) # Frame 2 (Alerts!)