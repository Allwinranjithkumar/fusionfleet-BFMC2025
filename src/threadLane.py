from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (serialCamera, SteerMotor, DrivingMode, kl, obcamera, ImuData, SpeedMotor)
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
        self.frame_widthh = None
        self.frame_height = None
        self.initial_midpoint = None
        self.initialcomp = None
        self.delta_comp = 0
        self.lane_history = {'left': [], 'right': []}
        self.smooth_factor = 5
        self.max_deviation = 100
        self.initial_position_set = False  # Flag to indicate if initial position is set
        self.subscribe()
        self.mohan = False
        
    def send(self):
        self.steerSender = messageHandlerSender(self.queuesList, SteerMotor)
        self.obSender = messageHandlerSender(self.queuesList, obcamera)
        self.motorspSender = messageHandlerSender(self.queuesList, SpeedMotor)
    
    def subscribe(self):
        self.serialCameraSubscriber = messageHandlerSubscriber(self.queuesList, serialCamera, "lastOnly", True)
        self.DrivingModeSubscriber = messageHandlerSubscriber(self.queuesList, DrivingMode, "lastOnly", True)
        self.klvalSubscriber = messageHandlerSubscriber(self.queuesList, kl, "lastOnly", True)
    
    def get_birds_eye_view(self, image):
        """Applies a perspective transform to get a bird's eye view of the track"""
        height, width = image.shape[:2]
        src_pts = np.float32([
            [width * 0.15, height * 0.9],  
            [width * 0.85, height * 0.9],  
            [width * 0.2, height * 0.35],  
            [width * 0.8, height * 0.35]   
        ])
        dst_pts = np.float32([
            [width * 0.2, height],
            [width * 0.8, height],
            [width * 0.2, 0],
            [width * 0.8, 0]
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (width, height))
        return warped
    
    def map_to_steering_angle(self, delta_comp):
            max_input = 60  # The max delta_comp value
            min_input = -60   # The min delta_comp value
            max_output = 25 # The max steering angle
            min_output = -25  # The min steering angle

            # Normalize delta_comp to range [-1, 1]
            normalized_delta = (2 * (delta_comp - min_input) / (max_input - min_input)) - 1  

            # Apply non-linear mapping (quadratic with sign)
            non_linear_factor = np.sign(normalized_delta) * (normalized_delta ** 2)

            # Scale to output range [-25, 25]
            steering_angle = np.clip(non_linear_factor * max_output, min_output, max_output)
        
            return steering_angle

    def detect_lanes(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        height, width = edges.shape
        roi_bottom_height = height // 2  
        vertices = np.array([[ 
            (0, height),  
            (0, roi_bottom_height),  
            (width, roi_bottom_height),   
            (width, height)  
        ]], dtype=np.int32)
        
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [vertices], 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=100)
        left_lines, right_lines = [], []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                if abs(slope) > 0.5:  
                    if slope < 0 and x1 < width / 2:
                        left_lines.append(line[0])
                    elif slope > 0 and x1 > width / 2:
                        right_lines.append(line[0])
        
        def get_average_line(lines):
            if len(lines) == 0:
                return None
            x1s = np.array([line[0] for line in lines])
            y1s = np.array([line[1] for line in lines])
            x2s = np.array([line[2] for line in lines])
            y2s = np.array([line[3] for line in lines])
            return [int(np.mean(x1s)), int(np.mean(y1s)), int(np.mean(x2s)), int(np.mean(y2s))]
        
        avg_left = get_average_line(left_lines)
        avg_right = get_average_line(right_lines)
        
        if (avg_left and avg_right) :
            midpoint_x = int((avg_left[0] + avg_right[0] + avg_left[2] + avg_right[2]) / 4)
            if not self.initial_position_set:
                self.initial_midpoint = midpoint_x
                self.initialcomp = midpoint_x
                self.initial_position_set = True
                return None
            delta_comp = midpoint_x - self.initialcomp
            
            return delta_comp
        elif avg_left:
                midpoint_x=int((avg_left[0] +avg_left[2]) / 2)
                delta_comp = midpoint_x - self.initialcomp
        elif avg_right:        
            delta_comp = midpoint_x - self.initialcomp
            return delta_comp
                    
        else:
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
                
                if yk == "hiiii":
                    mode = self.DrivingModeSubscriber.receive()
                    if mode in ["stop", "manual", "legacy"]:
                        self.send()
                        self.steerSender.send("0")
                        self.mohan = False
                        self.initial_position_set = False
                    if mode == "auto":
                        self.send()
                        self.mohan = True
                        self.initial_position_set = False
                
                if self.mohan and camera_data:
                    image_data = base64.b64decode(camera_data)
                    img = np.frombuffer(image_data, dtype=np.uint8)
                    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
                    send = not send
                    if frame is not None:
                        self.cap = frame
                        self.frame_widthh = frame.shape[1]
                        self.frame_height = frame.shape[0]
                        warped = self.get_birds_eye_view(frame)
                        delta_comp = self.detect_lanes(warped)
                        if delta_comp is not None:
                            steering_angle = self.map_to_steering_angle(delta_comp)
                            self.logging.info(f"Delta: {delta_comp}, Steering angle: {int(round(steering_angle))}")
                            self.steerSender.send(str(int(round(steering_angle)) * 10))
            except Exception as e:
                self.logging.error(f"Error in threadLane: {e}")

    def stop(self):
        self._running = False
        self.logging.info("Detection thread stopped successfully.")
