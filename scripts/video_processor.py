"""
Core video processing module for Flickd AI Engine
Handles frame extraction, object detection, and product matching
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from PIL import Image
import json
from pathlib import Path
import time

class VideoProcessor:
    def __init__(self):
        print("🎬 Initializing Video Processor...")
        
        # Load YOLOv8 model for fashion detection
        self.yolo_model = YOLO('yolov8n.pt')  # Using nano version for speed
        
        # Fashion item classes we're interested in
        self.fashion_classes = {
            'person': ['top', 'bottom', 'dress', 'jacket'],
            'handbag': ['bag'],
            'tie': ['accessory'],
            'suitcase': ['bag']
        }
        
        print(" Video Processor initialized")
    
    def extract_frames(self, video_path, max_frames=10):
        """Extract key frames from video"""
        print(f" Extracting frames from {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract frames at regular intervals
        interval = max(1, frame_count // max_frames)
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % interval == 0:
                frames.append({
                    'frame_number': frame_idx,
                    'image': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                })
                
            frame_idx += 1
            
            if len(frames) >= max_frames:
                break
        
        cap.release()
        print(f" Extracted {len(frames)} frames")
        return frames
    
    def detect_fashion_items(self, frames):
        """Detect fashion items in frames using YOLOv8"""
        print(" Detecting fashion items...")
        
        all_detections = []
        
        for frame_data in frames:
            frame = frame_data['image']
            frame_number = frame_data['frame_number']
            
            # Run YOLO detection
            results = self.yolo_model(frame)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class name
                        class_id = int(box.cls[0])
                        class_name = self.yolo_model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # Only process fashion-related items
                        if confidence > 0.5 and self.is_fashion_item(class_name):
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            detection = {
                                'frame_number': frame_number,
                                'class_name': class_name,
                                'confidence': confidence,
                                'bounding_box': {
                                    'x': int(x1),
                                    'y': int(y1),
                                    'w': int(x2 - x1),
                                    'h': int(y2 - y1)
                                },
                                'cropped_image': frame[int(y1):int(y2), int(x1):int(x2)]
                            }
                            all_detections.append(detection)
        
        print(f" Detected {len(all_detections)} fashion items")
        return all_detections
    
    def is_fashion_item(self, class_name):
        """Check if detected class is a fashion item"""
        fashion_keywords = ['person', 'handbag', 'tie', 'suitcase', 'backpack']
        return class_name.lower() in fashion_keywords
    
    def process_video(self, video_path):
        """Main processing pipeline for a single video"""
        start_time = time.time()
        
        print(f" Processing video: {video_path}")
        
        # Extract frames
        frames = self.extract_frames(video_path)
        
        # Detect fashion items
        detections = self.detect_fashion_items(frames)
        
        processing_time = time.time() - start_time
        
        result = {
            'video_path': str(video_path),
            'processing_time': round(processing_time, 2),
            'frames_processed': len(frames),
            'items_detected': len(detections),
            'detections': detections
        }
        
        print(f" Video processing complete in {processing_time:.2f}s")
        return result

# Example usage
if __name__ == "__main__":
    processor = VideoProcessor()
    
    # Mock video processing (since we don't have actual video files)
    print("🎬 Mock video processing demonstration")
    
    mock_result = {
        'video_path': '/data/videos/sample_video.mp4',
        'processing_time': 2.8,
        'frames_processed': 10,
        'items_detected': 3,
        'detections': [
            {
                'frame_number': 5,
                'class_name': 'person',
                'confidence': 0.89,
                'bounding_box': {'x': 120, 'y': 80, 'w': 200, 'h': 300}
            },
            {
                'frame_number': 8,
                'class_name': 'handbag',
                'confidence': 0.76,
                'bounding_box': {'x': 300, 'y': 150, 'w': 80, 'h': 120}
            }
        ]
    }
    
    print(" Mock processing result:")
    print(json.dumps(mock_result, indent=2))
