"""
Production Video Processor using YOLOv8
Handles frame extraction and fashion item detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from PIL import Image
import json
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)

class ProductionVideoProcessor:
    def __init__(self, model_size='n', confidence_threshold=0.5):
        """
        Initialize video processor with YOLOv8
        
        Args:
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
            confidence_threshold: Minimum confidence for detections
        """
        print(f"🎬 Initializing Production Video Processor...")
        
        self.confidence_threshold = confidence_threshold
        
        try:
            # Load YOLOv8 model
            model_path = f'yolov8{model_size}.pt'
            self.yolo_model = YOLO(model_path)
            print(f"YOLOv8{model_size} model loaded successfully")
            
        except Exception as e:
            print(f"📥 Downloading YOLOv8 model: {e}")
            self.yolo_model = YOLO('yolov8n.pt') 
        
        self.fashion_classes = {
            0: 'person',      
            24: 'handbag', 
            25: 'tie',     
            26: 'suitcase',  
            27: 'frisbee',    
            31: 'handbag',    
        }
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        print("Production Video Processor initialized")
    
    def extract_keyframes(self, video_path, max_frames=12):
        """Extract keyframes from video optimally"""
        print(f"Extracting keyframes from {Path(video_path).name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
        
        frames = []
        
        if duration <= 15:
            frame_interval = max(1, int(fps * 0.8))  
            frame_indices = list(range(0, total_frames, frame_interval))[:max_frames]
        else:
            frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = frame_idx / fps if fps > 0 else 0
                
                frames.append({
                    'frame_number': int(frame_idx),
                    'timestamp': round(timestamp, 2),
                    'image': frame_rgb,
                    'height': frame.shape[0],
                    'width': frame.shape[1]
                })
        
        cap.release()
        print(f"Extracted {len(frames)} keyframes")
        return frames
    
    def detect_fashion_items(self, frames):
        """Detect fashion items using YOLOv8"""
        print(f"Detecting fashion items...")
        
        all_detections = []
        
        for i, frame_data in enumerate(frames):
            frame = frame_data['image']
            frame_number = frame_data['frame_number']
            timestamp = frame_data['timestamp']
            
            try:
                results = self.yolo_model(frame, verbose=False)
                
                frame_detections = []
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            class_id = int(box.cls[0])
                            class_name = self.yolo_model.names[class_id]
                            confidence = float(box.conf[0])
                            if confidence >= self.confidence_threshold:
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                
                                #Calculate area and aspect ratio for filtering
                                width = x2 - x1
                                height = y2 - y1
                                area = width * height
                                aspect_ratio = width / height if height > 0 else 0
                                
                                #Filter out very small or oddly shaped detections
                                if area > 1000 and 0.2 < aspect_ratio < 5.0:
                                    cropped_region = frame[int(y1):int(y2), int(x1):int(x2)]
                                    
                                    detection = {
                                        'frame_number': frame_number,
                                        'timestamp': timestamp,
                                        'class_id': class_id,
                                        'class_name': class_name,
                                        'confidence': round(confidence, 3),
                                        'bounding_box': {
                                            'x': int(x1),
                                            'y': int(y1),
                                            'w': int(width),
                                            'h': int(height)
                                        },
                                        'cropped_image': cropped_region,
                                        'area': area,
                                        'aspect_ratio': round(aspect_ratio, 2)
                                    }
                                    
                                    frame_detections.append(detection)
                
                all_detections.extend(frame_detections)
                
            except Exception as e:
                logger.warning(f"Error processing frame {i+1}: {e}")
                continue
        
        filtered_detections = self.post_process_detections(all_detections)
        
        print(f"Detected {len(filtered_detections)} fashion items")
        return filtered_detections
    
    def post_process_detections(self, detections):
        """Post-process and filter detections"""
        if not detections:
            return []
        
        detections.sort(key=lambda x: (x['confidence'], x['area']), reverse=True)
        
        filtered = []
        for detection in detections:
            is_duplicate = False
            
            for existing in filtered:
                if (detection['frame_number'] == existing['frame_number'] and
                    detection['class_name'] == existing['class_name']):
                    
                    overlap = self.calculate_overlap(
                        detection['bounding_box'], 
                        existing['bounding_box']
                    )
                    
                    if overlap > 0.5:  # 50% overlap threshold
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered.append(detection)
        
        return filtered[:15]  # Max 15 detections per video
    
    def calculate_overlap(self, box1, box2):
        """Calculate overlap ratio between two bounding boxes"""
        x1_1, y1_1, w1, h1 = box1['x'], box1['y'], box1['w'], box1['h']
        x1_2, y1_2, w2, h2 = box2['x'], box2['y'], box2['w'], box2['h']
        
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        #Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def classify_fashion_type(self, detection):
        """Classify detection into fashion category"""
        class_name = detection['class_name'].lower()
        if class_name == 'person':
            aspect_ratio = detection['aspect_ratio']
            if aspect_ratio < 0.6:  # Tall and narrow - likely dress
                return 'dress'
            elif aspect_ratio > 1.5:  # Wide - likely top
                return 'top'
            else:
                return 'clothing'  # General clothing
        elif 'bag' in class_name or class_name in ['handbag', 'suitcase']:
            return 'bag'
        elif class_name == 'tie':
            return 'accessory'
        else:
            return 'accessory'
    
    def process_video(self, video_path):
        """Complete video processing pipeline"""
        start_time = time.time()
        video_path = Path(video_path)
        
        print(f"Processing video: {video_path.name}")
        
        try:
            # 1: Extract keyframes
            frames = self.extract_keyframes(video_path)
            
            if not frames:
                raise ValueError("No frames extracted from video")
            
            # 2: Detect fashion items
            detections = self.detect_fashion_items(frames)
            
            # 3: Classify fashion types
            for detection in detections:
                detection['fashion_type'] = self.classify_fashion_type(detection)
                detection['estimated_color'] = 'unknown' 
            
            processing_time = time.time() - start_time
            
            result = {
                'video_path': str(video_path),
                'video_name': video_path.name,
                'processing_time': round(processing_time, 2),
                'frames_processed': len(frames),
                'total_detections': len(detections),
                'detections': detections,
                'success': True
            }
            
            print(f"Video processing complete in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing video: {e}")
            
            return {
                'video_path': str(video_path),
                'video_name': video_path.name,
                'processing_time': round(processing_time, 2),
                'error': str(e),
                'success': False
            }

if __name__ == "__main__":
    processor = ProductionVideoProcessor()
    print("Production Video Processor ready")
    print("Place video files in /data/videos/ directory")
    print("Call processor.process_video(video_path) to process")