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
from mpu6050 import mpu6050

class threadRamp(ThreadWithStop):
    def init(self, queueList, logging, debugging=False):
        super().init()
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        self.cap = None
        self.frame_width = None
        self.frame_height = None
        self.incline_detected = False
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

    def read_pitch_angle(self):
        """Read pitch angle from IMU to detect inclines."""
        accel_data = self.sensor.get_accel_data()
        pitch = round(accel_data['y'], 3)  # 'y' axis represents front-back tilt
        return pitch

    def detect_ramp(self, pitch):
        """Detect ramp based on pitch angle."""
        uphill_threshold = 5.0   # Adjust based on sensor calibration
        downhill_threshold = -5.0

        if pitch > uphill_threshold:
            return "uphill"
        elif pitch < downhill_threshold:
            return "downhill"
        return "flat"

    def adjust_for_ramp(self, ramp_type):
        """Adjust vehicle behavior based on ramp type."""
        if ramp_type == "uphill":
            self.logging.info("Uphill detected! Increasing power.")
            self.motorspSender.send("700")  # Increase power
        elif ramp_type == "downhill":
            self.logging.info("Downhill detected! Reducing speed.")
            self.motorspSender.send("300")  # Reduce speed
        else:
            self.logging.info("Flat road detected. Maintaining normal speed.")
            self.motorspSender.send("500")  # Normal speed

    def process_frame(self, frame):
        """Process frame to detect lanes while handling ramps."""
        pitch_angle = self.read_pitch_angle()
        ramp_type = self.detect_ramp(pitch_angle)

        self.adjust_for_ramp(ramp_type)

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
                self.logging.error(f"Error in threadRamp: {e}")

    def stop(self):
        self._running = False
        self.logging.info("Ramp detection thread stopped successfully.")
