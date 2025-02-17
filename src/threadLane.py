from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (serialCamera, SteerMotor, DrivingMode, kl, obcamera,ImuData,SpeedMotor)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
import threading
import base64
import cv2
import time
import numpy as np
from mpu6050 import mpu6050


class threadLane(ThreadWithStop):
    def __init__(self, queueList, logging, debugging=False):
        super().__init__()
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        self.cap = None
        self.frame_width = None
        self.frame_height = None
        self.initial_midpoint = None
        self.initialcomp = None
        self.delta_comp = 0
        self.lane_history = {'left': [], 'right': []}
        self.smooth_factor = 5
        self.max_deviation= 100
        self.initial_position_set = False  # Flag to indicate if initial position is set
        self.subscribe()
        self.mohan = False
        

    def send(self):
        self.steerSender = messageHandlerSender(self.queuesList, SteerMotor)
        self.obSender = messageHandlerSender(self.queuesList, obcamera)
        self.imuDataSender = messageHandlerSender(self.queuesList, ImuData)
        self.motorspSender=messageHandlerSender(self.queuesList, SpeedMotor)
    def subscribe(self):
        self.serialCameraSubscriber = messageHandlerSubscriber(self.queuesList, serialCamera, "lastOnly", True)
        self.DrivingModeSubscriber = messageHandlerSubscriber(self.queuesList, DrivingMode, "lastOnly", True)
        self.klvalSubscriber = messageHandlerSubscriber(self.queuesList, kl, "lastOnly", True)
        self.sensor = mpu6050(0x68)
    def read_sensor_data(self):
        """Read accelerometer and gyroscope data from MPU6050"""
        accel_data = self.sensor.get_accel_data()
        gyro_data = self.sensor.get_gyro_data()
        data = {
            "roll": round(gyro_data['x'], 3),
            "pitch": round(gyro_data['y'], 3),
            "yaw": round(gyro_data['z'], 3),
            "accelx": round(accel_data['x'], 3),
            "accely": round(accel_data['y'], 3),
            "accelz": round(accel_data['z'], 3),
        }
        self.imuDataSender.send(str(data))
        return round(accel_data['x'], 3)

    def smooth_lanes(self, current_lane, lane_history):
        """Smooth detected lanes over multiple frames."""
        if current_lane is not None:
            lane_history.append(current_lane)
            if len(lane_history) > self.smooth_factor:
                lane_history.pop(0)
        else:
            if lane_history:
                lane_history.pop(0)

        if lane_history:
            avg_lane = np.mean(lane_history, axis=0)
            return tuple(map(int, avg_lane))
        return None

    def map_to_steering_angle(self, delta_comp):
        """Map lane deviation to steering angle."""
        max_delta = self.frame_width / 2
        
        return np.clip((delta_comp / max_delta) * 250, -25, 25)

    def process_frame(self, frame):
        """Process a single frame to detect lanes and compute steering angle."""
        self.frame_height, self.frame_width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        edges = cv2.Canny(blurred, 50, 150)

        roi_top = int(self.frame_height * 0.6)
        polygon = np.array([[(0, self.frame_height), (self.frame_width, self.frame_height),
                             (self.frame_width, roi_top), (0, roi_top)]], dtype=np.int32)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, polygon, 255)
        roi = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=30, minLineLength=50, maxLineGap=25)

        left_lines, right_lines = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else np.inf
                if slope < -0.5 and x1 < self.frame_width / 2 and x2 < self.frame_width / 2:
                    left_lines.append((x1, y1, x2, y2))
                elif slope > 0.5 and x1 > self.frame_width / 2 and x2 > self.frame_width / 2:
                    right_lines.append((x1, y1, x2, y2))

        left_lane = self.smooth_lanes(self.weighted_avg_lines(left_lines), self.lane_history['left'])
        right_lane = self.smooth_lanes(self.weighted_avg_lines(right_lines), self.lane_history['right'])

        lane_overlay = frame.copy()
        if left_lane:
            cv2.line(lane_overlay, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), (0, 255, 0), 5)
        if right_lane:
            cv2.line(lane_overlay, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), (0, 255, 0), 5)

        left_poly = self.fit_lane_line(left_lines) if left_lane else None
        right_poly = self.fit_lane_line(right_lines) if right_lane else None

        if left_poly is not None:
            left_x = np.polyval(left_poly, self.frame_height)
            if left_x > self.frame_width / 2:
                left_poly = None  

        if right_poly is not None:
            right_x = np.polyval(right_poly, self.frame_height)
            if right_x < self.frame_width / 2:
                right_poly = None  

        if left_poly is not None and right_poly is not None:
            left_x = np.polyval(left_poly, self.frame_height)
            right_x = np.polyval(right_poly, self.frame_height)

            if abs(left_x - right_x) < self.frame_width * 0.25:
                return lane_overlay, None

            midpoint_x = int(round((left_x + right_x) / 2))
        
        elif right_poly is not None:
            right_x = np.polyval(right_poly, self.frame_height)
            midpoint_x = int(round(right_x - self.frame_width * 0.35))
        else:
            return lane_overlay, None

        cv2.circle(lane_overlay, (midpoint_x, self.frame_height), 10, (0, 0, 255), -1)

        if not self.initial_position_set or abs(midpoint_x - self.initial_midpoint) > self.max_deviation:
            self.initial_midpoint = midpoint_x
            self.initialcomp = midpoint_x
            self.initial_position_set = True
            return lane_overlay, None

        self.delta_comp = midpoint_x - self.initialcomp
        return lane_overlay, self.delta_comp

    def weighted_avg_lines(self, lines):
        """Compute weighted average of detected lane lines."""
        if not lines:
            return None

        weights = [np.sqrt((x2 - x1)**2 + (y2 - y1)**2) for x1, y1, x2, y2 in lines]
        total_weight = sum(weights)

        if total_weight == 0:
            return None

        avg_x1 = int(sum(x1 * w for (x1, _, _, _), w in zip(lines, weights)) / total_weight)
        avg_y1 = int(sum(y1 * w for (_, y1, _, _), w in zip(lines, weights)) / total_weight)
        avg_x2 = int(sum(x2 * w for (_, _, x2, _), w in zip(lines, weights)) / total_weight)
        avg_y2 = int(sum(y2 * w for (_, _, _, y2), w in zip(lines, weights)) / total_weight)

        return avg_x1, avg_y1, avg_x2, avg_y2

    def fit_lane_line(self, lines):
        """Fit a polynomial to the detected lane lines."""
        if not lines:
            return None

        points = [(x1, y1) for x1, y1, _, _ in lines] + [(x2, y2) for _, _, x2, y2 in lines]
        x_vals = np.array([p[0] for p in points])
        y_vals = np.array([p[1] for p in points])

        if len(x_vals) >= 2:
            return np.polyfit(y_vals, x_vals, 1)
        return None

    def run(self):
        send = False
        while self._running:
            try:
                camera_data = self.serialCameraSubscriber.receive()
                if camera_data and send:
                    self.send()
                    self.obSender.send(camera_data)
                yk = self.klvalSubscriber.receive()

                if yk is not None:
                    if yk == "hiiii":
                        mode = self.DrivingModeSubscriber.receive()
                        if mode in ["stop", "manual", "legacy"]:
                            self.send()
                            self.steerSender.send("0")
                            self.mohan = False
                            self.initial_position_set = False
                        if mode == "auto":
                            self.send()
                            self.mohan = False
                            self.initial_position_set = False
                            self.mohan = True

                if self.mohan:
                    if camera_data:
                        image_data = base64.b64decode(camera_data)
                        img = np.frombuffer(image_data, dtype=np.uint8)
                        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
                        send=not send
                        imu=self.read_sensor_data()
                        if frame is not None:
                            self.cap = frame
                            self.frame_width = frame.shape[1]
                            self.frame_height = frame.shape[0]
                            lane_overlay,delta_comp = self.process_frame(frame)
                        if imu < -0.850: 
                            #self.motorspSender.send("150")
                            self.logging.info(imu)
                        if (imu > 1.300 and imu <2.5):
                            self.logging.info(imu)
                            #self.motorspSender.send("500")
                        if delta_comp is not None:
                                steering_angle = self.map_to_steering_angle(delta_comp)
                                steering_angle_int = int(round(steering_angle))
                                self.logging.info(f"Delta: {delta_comp}, Steering angle (rounded): {steering_angle_int}")
                                #self.steerSender.send(str(steering_angle_int*10))
            except Exception as e:
                self.logging.error(f"Error in threadLane: {e}")

    def stop(self):
        self._running = False
        self.logging.info("Detection thread stopped successfully.")
