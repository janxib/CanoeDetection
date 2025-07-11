#!/usr/bin/env python3


import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")


class FilteredDetector:
    """YOLO detector with class filtering and configuration support"""
    
    def __init__(self, model_path: str = "yolov8n.pt", config_path: Optional[str] = None):
        """Initialize the filtered detector"""
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")
        
        self.model_path = model_path
        self.model = None
        self.class_names = {}
        self.config = self._load_config(config_path)
        self._load_model()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "target_classes": {
                8: {"name": "boat", "confidence": 0.4, "color": [0, 255, 0]},
                37: {"name": "surfboard", "confidence": 0.5, "color": [0, 255, 255]},
                0: {"name": "person", "confidence": 0.6, "color": [255, 0, 0]},
                32: {"name": "sports ball", "confidence": 0.7, "color": [255, 255, 0]},
                29: {"name": "frisbee", "confidence": 0.6, "color": [255, 0, 255]}
            },
            "detection": {
                "base_confidence": 0.3,
                "iou_threshold": 0.5,
                "max_detections": 50
            },
            "visualization": {
                "show_confidence": True,
                "show_class_name": True,
                "box_thickness": 2
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge configurations
                for key, value in user_config.items():
                    if key in default_config:
                        if isinstance(default_config[key], dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
                    else:
                        default_config[key] = value
        
        return default_config
    
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
                    32: 'sports ball', 33: 'kite', 37: 'surfboard', 29: 'frisbee'
                }
            
            print(f"Model loaded successfully!")
            print(f"Target classes: {list(self.config['target_classes'].keys())}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect and filter specific object classes"""
        if self.model is None:
            return []
        
        # Run YOLO detection
        results = self.model(
            frame, 
            conf=self.config['detection']['base_confidence'],
            iou=self.config['detection']['iou_threshold'],
            max_det=self.config['detection']['max_detections'],
            verbose=False
        )
        
        detections = []
        target_classes = self.config['target_classes']
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    # Check if this class is in our target list
                    if cls_id not in target_classes:
                        continue
                    
                    # Check class-specific confidence threshold
                    class_info = target_classes[cls_id]
                    if conf < class_info['confidence']:
                        continue
                    
                    x1, y1, x2, y2 = box
                    class_name = class_info['name']
                    
                    detection = {
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(conf),
                        'class_id': cls_id,
                        'class_name': class_name,
                        'color': class_info['color'],
                        'width': int(x2 - x1),
                        'height': int(y2 - y1),
                        'area': int((x2 - x1) * (y2 - y1))
                    }
                    detections.append(detection)
        
        return detections

    def apply_geometric_filters(self, detections: List[Dict]) -> List[Dict]:
        """Apply basic geometric filtering"""
        filtered_detections = []
        
        for detection in detections:
            width = detection['width']
            height = detection['height']
            area = detection['area']
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1.0
            
            # Basic size filters
            if width < 30 or height < 15:  # Too small
                continue
            if width > 800 or height > 400:  # Too large
                continue
            if area < 500:  # Too small area
                continue
            
            # Aspect ratio filters for water-related objects
            if detection['class_id'] in [8, 37]:  # boats, surfboards
                if aspect_ratio < 1.5 or aspect_ratio > 15:  # Should be elongated
                    continue
            
            # Add geometric info to detection
            detection['aspect_ratio'] = aspect_ratio
            filtered_detections.append(detection)
        
        return filtered_detections

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw filtered detections with enhanced visualization"""
        annotated_frame = frame.copy()
        viz_config = self.config['visualization']
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            color = tuple(detection['color'])
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, viz_config['box_thickness'])
            
            # Prepare label
            label_parts = []
            if viz_config['show_class_name']:
                label_parts.append(class_name)
            if viz_config['show_confidence']:
                label_parts.append(f"{confidence:.2f}")
            
            # Add geometric info
            label_parts.append(f"{detection['width']}x{detection['height']}")
            label_parts.append(f"AR:{detection['aspect_ratio']:.1f}")
            
            label = " ".join(label_parts)
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10), 
                         (x1 + label_width + 5, y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1 + 2, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(annotated_frame, (center_x, center_y), 3, color, -1)
        
        return annotated_frame

    def process_video(self, video_path: str, output_path: str = None, show_live: bool = False):
        """Process video with filtered detection"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n=== Week 2: Filtered Detection ===")
        print(f"Video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        print(f"Target classes: {len(self.config['target_classes'])}")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_number = 0
        total_detections = 0
        class_counts = {}
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect objects
                detections = self.detect_objects(frame)
                
                # Apply geometric filters
                filtered_detections = self.apply_geometric_filters(detections)
                total_detections += len(filtered_detections)
                
                # Count detections by class
                for detection in filtered_detections:
                    class_name = detection['class_name']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                # Draw detections
                annotated_frame = self.draw_detections(frame, filtered_detections)
                
                # Add frame info
                info_lines = [
                    f"Week 2: Filtered Detection",
                    f"Frame: {frame_number}/{total_frames}",
                    f"Detections: {len(filtered_detections)}"
                ]
                
                for i, line in enumerate(info_lines):
                    cv2.putText(annotated_frame, line, (10, 25 + i * 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Show live preview
                if show_live:
                    cv2.imshow('Week 2: Filtered Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Write frame
                if writer:
                    writer.write(annotated_frame)
                
                frame_number += 1
                
                # Progress update
                if frame_number % 30 == 0:
                    progress = frame_number / total_frames * 100
                    print(f"Progress: {progress:.1f}% - Total detections: {total_detections}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_live:
                cv2.destroyAllWindows()
        
        print(f"Total detections: {total_detections}")
        print(f"Average detections per frame: {total_detections/frame_number:.2f}")
        print(f"Detection breakdown:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Week 2: Filtered Object Detection')
    
    parser.add_argument('--video', '-v', required=True, help='Path to input video file')
    parser.add_argument('--model', '-m', default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--live', '-l', action='store_true', help='Show live preview')
    
    args = parser.parse_args()
    
    if not YOLO_AVAILABLE:
        print("Error: ultralytics package required. Install with: pip install ultralytics")
        return 1
    
    try:
        # Initialize detector
        detector = FilteredDetector(args.model, args.config)
        
        # Process video
        detector.process_video(args.video, args.output, args.live)
        
        print("Week 2 challenge completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 