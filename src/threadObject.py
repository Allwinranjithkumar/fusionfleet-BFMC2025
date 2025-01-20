import threading
import cv2
import time
import numpy as np
import base64
import torch
from ultralytics import YOLO
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import serialCamera, DrivingMode, kl, SpeedMotor,WarningSignal,obcamera
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
       
        self.motorspSender = None
        self.var=False
        
       # Delay before starting frame processing

    def load_model(self):
        """Load the object detection model."""
        self.model = YOLO("/home/raspi/Bosch/Brain/src/Detection/Object/objdet.pt")
        self.model_input_shape = (512,270)

    def subscribe(self):
        """Subscribe to required message channels."""
        self.serialCameraSubscriber = messageHandlerSubscriber(self.queuesList, obcamera, "lastOnly", True)
        self.DrivingModeSubscriber = messageHandlerSubscriber(self.queuesList, DrivingMode, "lastOnly", True)
        self.HiSubscriber=messageHandlerSubscriber(self.queuesList, kl, "lastOnly",True)
    
    def send(self):
         self.motorspSender = messageHandlerSender(self.queuesList, SpeedMotor)
         self.wa = messageHandlerSender(self.queuesList, WarningSignal)
         
        
             # Use self.queuesList

    
    
    
    
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

    def is_object_detected(self, results):
        try:
            for result in results:
                for box in result.boxes:
                    conf = box.conf.item() if hasattr(box.conf, 'item') else box.conf  # Access confidence
                    class_id = int(box.cls) if hasattr(box, 'cls') else box.cls
                    detected_class = self.model.names[class_id] if class_id in self.model.names else "unknown"
                    
                    self.logging.info(f"Detected class: {detected_class}, confidence: {conf:.2f}")
                    
                    if conf > 0.5 and detected_class == 'stop-sign':
                        self.motorspSender.send("0")
                        
                        return True
                    if conf > 0.5 and detected_class ==  'stop-line':
                        self.motorspSender.send("0")
                       
                        return True
                    if conf > 0.5 and detected_class == 'traffic-light':
                        self.motorspSender.send("20")
                        return True   
                    if conf > 0.5 and detected_class == 'crosswalk-sign': 
                        self.motorspSender.send("30")
                        return True    
                    if conf > 0.5 and detected_class ==  'pedestrian':
                        self.motorspSender.send("40")
                        return True 
                    if conf > 0.5 and detected_class ==  'priority-sign': 
                        self.motorspSender.send("50")
                        return True
                    if conf > 0.5 and detected_class ==  'round-about-sign': 
                        self.motorspSender.send("60")
                        return True
                    if conf > 0.5 and detected_class ==  'one-way-road-sign':
                        self.motorspSender.send("70")
                        return True
                    if conf > 0.5 and detected_class ==  'no-entry-road-sign': 
                        self.motorspSender.send("80")
                        return True
                    if conf > 0.5 and detected_class == 'parking-sign':
                        self.motorspSender.send("90")
                        
                        return True  
                    if conf > 0.5 and detected_class ==  'parking-spot':
                        self.motorspSender.send("110")
                    
                        return True 
                    if conf > 0.5 and detected_class ==  'highway-entry-sign':
                        self.motorspSender.send("120")
                        return True 
                    if conf > 0.5 and detected_class ==  'highway-exit-sign': 
                        self.motorspSender.send("130")
                        return True 
                    if conf > 0.5 and detected_class ==  'closed-road-stand': 
                        self.motorspSender.send("140")
                        return True   
                    if conf > 0.5 and detected_class ==  'car': 
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
    
                            self.send()
                            
                            self.motorspSender.send("0")
                            self.var=False
                        if mode=="auto":
                            self.send()
                            
                            self.motorspSender.send("200")
                            self.var=True
                    
                if self.var:
                    if camera_data:
                        image_data = base64.b64decode(camera_data)
                        img = np.frombuffer(image_data, dtype=np.uint8)
                        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
                    if frame is not None:
                        preprocessed_frame = self.preprocess_frame(frame)
                        detection_results = self.model_infer(preprocessed_frame)
                        if self.is_object_detected(detection_results):
                            self.logging.info("Object detected.")
                            
                        else:
                            self.logging.info("No object detected.")
                            self.motorspSender.send("200")
                            
                            
                    else:
                        self.logging.error("Decoded frame is None.")
      
                else:
                    pass
            except Exception as e:
                self.logging.error(f"Error in main thread loop: {e}")

    def stop(self):
        """Stop the detection thread."""
        self._running = False
        self.logging.info("Detection thread stopped successfully.")
