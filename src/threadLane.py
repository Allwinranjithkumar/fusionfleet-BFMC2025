from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (serialCamera, SteerMotor, DrivingMode, kl, obcamera)
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender
import threading
import base64
import cv2
import time
import numpy as np

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
        self.initial_position_set = False  # Flag to indicate if initial position is set
        self.subscribe()
        self.mohan = False
        

    def send(self):
        self.steerSender = messageHandlerSender(self.queuesList, SteerMotor)
        self.obSender = messageHandlerSender(self.queuesList, obcamera)

    def subscribe(self):
        self.serialCameraSubscriber = messageHandlerSubscriber(self.queuesList, serialCamera, "lastOnly", True)
        self.DrivingModeSubscriber = messageHandlerSubscriber(self.queuesList, DrivingMode, "lastOnly", True)
        self.klvalSubscriber = messageHandlerSubscriber(self.queuesList, kl, "lastOnly", True)

    def smooth_lanes(self, current_lane, lane_history):
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
        max_delta = self.frame_width / 2  # Assume max delta_comp is half the frame width
        steering_angle = np.clip((delta_comp / max_delta) * 250, -25, 25)
        return steering_angle

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        edges = cv2.Canny(blurred, 15, 100)

        roi_top = int(self.frame_height * 0.6)
        polygon = np.array([[(0, self.frame_height), (self.frame_width, self.frame_height), (self.frame_width, roi_top), (0, roi_top)]], dtype=np.int32)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, polygon, 255)
        roi = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=30, minLineLength=50, maxLineGap=25)

        left_lines, right_lines = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else np.inf
                if slope < -0.5:
                    left_lines.append((x1, y1, x2, y2))
                elif slope > 0.5:
                    right_lines.append((x1, y1, x2, y2))

        left_lane = self.smooth_lanes(self.weighted_avg_lines(left_lines), self.lane_history['left'])
        right_lane = self.smooth_lanes(self.weighted_avg_lines(right_lines), self.lane_history['right'])

        left_poly = self.fit_lane_line(left_lines) if left_lane is not None else None
        right_poly = self.fit_lane_line(right_lines) if right_lane is not None else None

        if left_poly is not None and right_poly is not None:
            midpoint_x = int(round((np.polyval(left_poly, self.frame_height) + np.polyval(right_poly, self.frame_height)) / 2))

            if not self.initial_position_set:
                self.initial_midpoint = midpoint_x
                self.initialcomp = midpoint_x
                self.initial_position_set = True
                self.logging.info(f"Initial midpoint x-coordinate set: {self.initialcomp}")
                return None

            self.delta_comp =  midpoint_x-self.initialcomp 
            return self.delta_comp
        return None

    def weighted_avg_lines(self, lines):
        if len(lines) == 0:
            return None

        weights = []
        for line in lines:
            length = np.sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2)
            weights.append(length)

        total_weight = sum(weights)
        normalized_weights = [weight / total_weight for weight in weights]

        avg_x1 = int(sum([line[0] * weight for line, weight in zip(lines, normalized_weights)]))
        avg_y1 = int(sum([line[1] * weight for line, weight in zip(lines, normalized_weights)]))
        avg_x2 = int(sum([line[2] * weight for line, weight in zip(lines, normalized_weights)]))
        avg_y2 = int(sum([line[3] * weight for line, weight in zip(lines, normalized_weights)]))

        return (avg_x1, avg_y1, avg_x2, avg_y2)

    def fit_lane_line(self, lines):
        if len(lines) == 0:
            return None

        points = []
        for line in lines:
            x1, y1, x2, y2 = line
            points.append((x1, y1))
            points.append((x2, y2))

        x_vals = np.array([point[0] for point in points])
        y_vals = np.array([point[1] for point in points])

        if len(x_vals) >= 2:
            poly_coeff = np.polyfit(y_vals, x_vals, 1)
            return poly_coeff
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
                        if frame is not None:
                            self.cap = frame
                            self.frame_width = frame.shape[1]
                            self.frame_height = frame.shape[0]
                            delta_comp = self.process_frame(frame)
                            if delta_comp is not None:
                                steering_angle = self.map_to_steering_angle(delta_comp)
                                steering_angle_int = int(round(steering_angle))
                                self.logging.info(f"Delta: {delta_comp}, Steering angle (rounded): {steering_angle_int}")
                                self.steerSender.send(str(steering_angle_int*10))
            except Exception as e:
                self.logging.error(f"Error in threadLane: {e}")

    def stop(self):
        self._running = False
        self.logging.info("Detection thread stopped successfully.")
