#!/usr/bin/env python3
"""
Week 1: Basic YOLO Object Detection
===================================
Learn the fundamentals of YOLO object detection
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")


class BasicYOLODetector:
    """Basic YOLO object detector for learning fundamentals"""
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        """Initialize the basic detector"""
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")
        
        self.model_path = model_path
        self.model = None
        self.class_names = {}
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            print(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
            else:
                # COCO class names
                self.class_names = {
                    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
                    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
                    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
                    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
                    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket'
                }
            
            print(f"Model loaded successfully!")
            print(f"Available classes: {len(self.class_names)}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def detect_objects(self, frame: np.ndarray, confidence: float = 0.5):
        """Detect objects in a frame"""
        if self.model is None:
            return []
        
        # Run YOLO detection
        results = self.model(frame, conf=confidence, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = box
                    class_name = self.class_names.get(cls_id, 'unknown')
                    
                    detection = {
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(conf),
                        'class_id': cls_id,
                        'class_name': class_name
                    }
                    detections.append(detection)
        
        return detections

    def draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw detection boxes and labels on frame"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Choose color based on class
            if detection['class_id'] == 8:  # boat
                color = (0, 255, 0)  # Green
            elif detection['class_id'] == 0:  # person
                color = (255, 0, 0)  # Blue
            else:
                color = (0, 0, 255)  # Red
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{class_name}: {confidence:.2f}"
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10), 
                         (x1 + label_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame

    def process_video(self, video_path: str, output_path: str = None, show_live: bool = False):
        """Process video file"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n=== Week 1: Basic YOLO Detection ===")
        print(f"Video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_number = 0
        total_detections = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect objects
                detections = self.detect_objects(frame, confidence=0.5)
                total_detections += len(detections)
                
                # Draw detections
                annotated_frame = self.draw_detections(frame, detections)
                
                # Add frame info
                info_text = f"Frame: {frame_number}/{total_frames}, Detections: {len(detections)}"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Show live preview
                if show_live:
                    cv2.imshow('Week 1: Basic YOLO Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Write frame
                if writer:
                    writer.write(annotated_frame)
                
                frame_number += 1
                
                # Progress update
                if frame_number % 30 == 0:
                    progress = frame_number / total_frames * 100
                    print(f"Progress: {progress:.1f}% - Detections so far: {total_detections}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_live:
                cv2.destroyAllWindows()
        
        print(f"\n=== Week 1 Complete ===")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per frame: {total_detections/frame_number:.2f}")


def main():
    """Main function for Week 1"""
    parser = argparse.ArgumentParser(description='Week 1: Basic YOLO Object Detection')
    
    parser.add_argument('--video', '-v', required=True, help='Path to input video file')
    parser.add_argument('--model', '-m', default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--live', '-l', action='store_true', help='Show live preview')
    
    args = parser.parse_args()
    
    if not YOLO_AVAILABLE:
        print("Error: ultralytics package required. Install with: pip install ultralytics")
        return 1
    
    try:
        # Initialize detector
        detector = BasicYOLODetector(args.model)
        
        # Process video
        detector.process_video(args.video, args.output, args.live)
        
        print("Week 1 challenge completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 