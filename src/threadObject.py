import threading
import cv2
import time
import numpy as np
import base64
import torch
from ultralytics import YOLO
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import serialCamera, DrivingMode, kl, SpeedMotor,WarningSignal,obcamera,SteerMotor,rd
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender



class threadObject(ThreadWithStop):
    def __init__(self, queueList, logging, debugging=False):
        super().__init__()
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        self.load_model()
        self.subscribe()
        self.count=0
        self.motorspSender = None
        self.var=False
        self.rd=None
        self.hi=0
       # Delay before starting frame processing
        self.stop=0
    def load_model(self):
        """Load the object detection model."""
        self.model = YOLO("/home/raspi/Brain/src/Detection/Object/trafficnew.pt")
        self.model_input_shape = (640,640)

    def subscribe(self):
        """Subscribe to required message channels."""
        self.serialCameraSubscriber = messageHandlerSubscriber(self.queuesList, obcamera, "lastOnly", True)
        self.DrivingModeSubscriber = messageHandlerSubscriber(self.queuesList, DrivingMode, "lastOnly", True)
        self.HiSubscriber=messageHandlerSubscriber(self.queuesList, kl, "lastOnly",True)
        self.start_time = time.time() 
    def send(self):
         self.motorspSender = messageHandlerSender(self.queuesList, SpeedMotor)
         self.wa = messageHandlerSender(self.queuesList, WarningSignal)
         self.steerSender = messageHandlerSender(self.queuesList, SteerMotor)
         self.rdSender = messageHandlerSender(self.queuesList, rd)
         # Use self.queuesLis
    
    def preprocess_frame(self, frame):
        """Preprocess the frame for inference."""
        
        return cv2.resize(frame, self.model_input_shape)

    def model_infer(self, frame):
        """Run inference on the preprocessed frame."""
        try:
            results = self.model(frame)
            return results
        except Exception as e:
            
            return []

    def is_object_detected(self, results,frame):
        try:
            detected = False
            for result in results:
                for box in result.boxes:
                    conf = box.conf.item() if hasattr(box.conf, 'item') else box.conf  # Access confidence
                    class_id = int(box.cls) if hasattr(box, 'cls') else box.cls
                    detected_class = self.model.names[class_id] if class_id in self.model.names else "unknown"
                    (x1, y1, x2, y2) = box.xyxy[0].tolist()  # Get bounding box coordinates
                    self.logging.info(f"Detected class: {detected_class}, confidence: {conf:.2f}")
                    if conf > 0.5 :
                        color = (0, 255, 0)  # Green for detected objects
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        label = f"{detected_class} ({conf:.2f})"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        detected = True
                        if conf > 0.5 and detected_class == 'roundabout':
                               self.steerSender("-25")
                               return True
                        if conf > 0.5 and (detected_class == 'stop-sign' or detected_class == 'Stop'):
                               self.motorspSender.send("0")
                               self.stop=1
                               return True
                        if conf > 0.5 and detected_class == 'crosswalk-sign':
                               self.motorspSender.send("50")
                               return True 
                        if conf > 0.5 and (detected_class == 'priority-sign' or detected_class == 'Priority road'):
                                   self.motorspSender.send("0")
                                   self.hi=1
                                   return True
                        if conf > 0.5 and (detected_class == 'Motorway' or detected_class == 'onHighway'):
                                   self.motorspSender.send("0")
                              
                                   return True  
                        if conf > 0.5 and (detected_class == 'parking' ):
                                   self.motorspSender.send("300")
                                   self.steerSender("-25")
                                   self.motorspSender.send("300")
                                   self.steerSender("0")
                                   self.motorspSender.send("0")
                                   return True                      
                                   
                        if conf > 0.5 and (detected_class == 'oneway' ):
                                   self.motorspSender.send("0")
                                   return True                                      
                            
                                     
                        if conf > 0.5 and (detected_class == 'Noentry' ):
                                   self.steerSender("-25")
                                   return True   
                                   
                        if conf > 0.5 and detected_class == 'offHighway':
                                   self.motorspSender.send("150")
                                   return True                      
                    return False
        except Exception as e:
            self.logging.error(f"Error processing detection results: {e}")
            return False
    def run(self):
        """Run the detection process after an initial delay."""
        while self._running:
            try:
                camera_data = self.serialCameraSubscriber.receive()
                yk=self.HiSubscriber.receive()
                if yk is not None:
                    if yk == "hiiii":
                        mode=self.DrivingModeSubscriber.receive()
                        if mode=="stop" or mode=="manual" or mode=="legacy":
                            self.count=0
                            self.send()
                            self.steerSender.send("0")
                            self.motorspSender.send("0")
                            self.var=False
                        if mode=="auto":
                            self.count=0
                            self.send()
                            self.motorspSender.send("150")
                            self.steerSender.send("0")
                            self.var=True
                if self.var:
                    if camera_data:
                        image_data = base64.b64decode(camera_data)
                        img = np.frombuffer(image_data, dtype=np.uint8)
                        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
                    if frame is not None:
                        preprocessed_frame = self.preprocess_frame(frame)
                        detection_results = self.model_infer(preprocessed_frame)
                        if self.is_object_detected(detection_results,frame):
                            self.logging.info("Object detected.")
                        if self.stop==1:
                            self.steerSender.send("0")
                            self.motorspSender.send("0") 
                            
                        if self.hi==1:
                            self.motorspSender.send("0")
                            self.hi=0

                        else:
                            self.logging.info("No object detected.")
                            self.steerSender.send("0")
                            self.motorspSender.send("150")
                            
                    else:
                      self.logging.error("Decoded frame is None.")
                time.sleep(0.05)

            except Exception as e:
                self.logging.error(f"Error in main thread loop: {e}")
          
    def stop(self):
        """Stop the detection thread."""
        self._running = False
        self.logging.info("Detection thread stopped successfully.")
