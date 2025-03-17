from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (
    serialCamera, SteerMotor, DrivingMode, kl, obcamera, ImuData, SpeedMotor
)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
import threading
import base64
import cv2
import time
import numpy as np


class threadTunnel(ThreadWithStop):
    def init(self, queueList, logging, debugging=False):
        super().init()
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        self.cap = None
        self.frame_width = None
        self.frame_height = None
        self.in_tunnel = False  # Track tunnel state
        self.subscribe()

    def send(self):
        self.steerSender = messageHandlerSender(self.queuesList, SteerMotor)
        self.obSender = messageHandlerSender(self.queuesList, obcamera)
        self.imuDataSender = messageHandlerSender(self.queuesList, ImuData)
        self.motorspSender = messageHandlerSender(self.queuesList, SpeedMotor)

    def subscribe(self):
        self.serialCameraSubscriber = messageHandlerSubscriber(self.queuesList, serialCamera, "lastOnly", True)
        self.DrivingModeSubscriber = messageHandlerSubscriber(self.queuesList, DrivingMode, "lastOnly", True)
        self.klvalSubscriber = messageHandlerSubscriber(self.queuesList, kl, "lastOnly", True)
        self.sensor = mpu6050(0x68)

    def detect_tunnel(self, frame):
        """Detect tunnel by analyzing frame brightness."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)

        tunnel_threshold = 40  # Adjust based on lighting conditions
        return avg_brightness < tunnel_threshold

    def adjust_for_tunnel(self):
        """Adjust vehicle behavior when in a tunnel."""
        self.logging.info("Tunnel detected! Reducing speed and adjusting steering.")
        self.motorspSender.send("200")  # Reduce speed
        # Optionally, enable infrared mode or headlights

    def adjust_after_tunnel(self):
        """Reset vehicle behavior after exiting a tunnel."""
        self.logging.info("Exited tunnel. Restoring normal speed.")
        self.motorspSender.send("500")  # Restore normal speed

    def process_frame(self, frame):
        """Process frame to detect tunnels and adjust navigation."""
        if self.detect_tunnel(frame):
            if not self.in_tunnel:
                self.in_tunnel = True
                self.adjust_for_tunnel()
        else:
            if self.in_tunnel:
                self.in_tunnel = False
                self.adjust_after_tunnel()

    def run(self):
        while self._running:
            try:
                camera_data = self.serialCameraSubscriber.receive()
                if camera_data:
                    image_data = base64.b64decode(camera_data)
                    img = np.frombuffer(image_data, dtype=np.uint8)
                    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

                    if frame is not None:
                        self.process_frame(frame)

            except Exception as e:
                self.logging.error(f"Error in threadTunnel: {e}")

    def stop(self):
        self._running = False
        self.logging.info("Tunnel detection thread stopped successfully.")
