#!/usr/bin/env python3


import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")


@dataclass
class TrackedObject:
    """Data class for tracked objects"""
    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    centroid: Tuple[float, float]
    age: int = 0
    max_age: int = 20
    color: Tuple[int, int, int] = (0, 255, 0)
    history: List[Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []


class TrackingDetector:
    """YOLO detector with object tracking capabilities"""
    
    def __init__(self, model_path: str = "yolov8n.pt", config_path: Optional[str] = None):
        """Initialize the tracking detector"""
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")
        
        self.model_path = model_path
        self.model = None
        self.class_names = {}
        self.config = self._load_config(config_path)
        self._load_model()
        
        # Tracking variables
        self.active_tracks: Dict[int, TrackedObject] = {}
        self.next_track_id = 0
        self.max_distance = 100  # Maximum distance for track association
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "target_classes": {
                8: {"name": "boat", "confidence": 0.4, "color": [0, 255, 0]},
                37: {"name": "surfboard", "confidence": 0.5, "color": [0, 255, 255]},
                0: {"name": "person", "confidence": 0.6, "color": [255, 0, 0]}
            },
            "detection": {
                "base_confidence": 0.3,
                "iou_threshold": 0.5,
                "max_detections": 50
            },
            "tracking": {
                "max_distance": 100,
                "max_age": 20,
                "min_confidence": 0.3
            },
            "visualization": {
                "show_tracks": True,
                "show_trail": True,
                "trail_length": 30,
                "show_track_id": True
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
        
        self.max_distance = default_config["tracking"]["max_distance"]
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
                    0: 'person', 8: 'boat', 37: 'surfboard'
                }
            
            print(f"Model loaded successfully!")
            print(f"Target classes: {list(self.config['target_classes'].keys())}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects in frame"""
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
                    centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                    
                    detection = {
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(conf),
                        'class_id': cls_id,
                        'class_name': class_info['name'],
                        'color': class_info['color'],
                        'centroid': centroid
                    }
                    detections.append(detection)
        
        return detections

    def calculate_distance(self, centroid1: Tuple[float, float], centroid2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two centroids"""
        return np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)

    def update_tracks(self, detections: List[Dict]) -> List[TrackedObject]:
        """Update object tracks with new detections"""
        # Create list of current detections
        current_detections = []
        for detection in detections:
            current_detections.append(detection)
        
        # Track assignment using Hungarian algorithm (simplified version)
        tracked_objects = []
        used_detections = set()
        
        # Try to match existing tracks
        for track_id, track in list(self.active_tracks.items()):
            best_detection_idx = None
            best_distance = float('inf')
            
            # Find closest detection
            for i, detection in enumerate(current_detections):
                if i in used_detections:
                    continue
                
                if detection['class_id'] != track.class_id:
                    continue
                
                distance = self.calculate_distance(track.centroid, detection['centroid'])
                
                if distance < best_distance and distance < self.max_distance:
                    best_distance = distance
                    best_detection_idx = i
            
            # Update track if match found
            if best_detection_idx is not None:
                detection = current_detections[best_detection_idx]
                used_detections.add(best_detection_idx)
                
                # Update track
                track.bbox = detection['bbox']
                track.confidence = detection['confidence']
                track.centroid = detection['centroid']
                track.age = 0  # Reset age
                
                # Update history
                track.history.append(track.centroid)
                if len(track.history) > self.config['visualization']['trail_length']:
                    track.history.pop(0)
                
                tracked_objects.append(track)
            else:
                # Age the track
                track.age += 1
                if track.age <= track.max_age:
                    tracked_objects.append(track)
                else:
                    # Remove old track
                    del self.active_tracks[track_id]
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(current_detections):
            if i not in used_detections:
                new_track = TrackedObject(
                    track_id=self.next_track_id,
                    class_id=detection['class_id'],
                    class_name=detection['class_name'],
                    bbox=detection['bbox'],
                    confidence=detection['confidence'],
                    centroid=detection['centroid'],
                    color=tuple(detection['color']),
                    max_age=self.config['tracking']['max_age']
                )
                new_track.history = [new_track.centroid]
                
                self.active_tracks[self.next_track_id] = new_track
                tracked_objects.append(new_track)
                self.next_track_id += 1
        
        return tracked_objects

    def draw_tracks(self, frame: np.ndarray, tracked_objects: List[TrackedObject]) -> np.ndarray:
        """Draw tracked objects with trails and IDs"""
        annotated_frame = frame.copy()
        viz_config = self.config['visualization']
        
        for track in tracked_objects:
            x1, y1, x2, y2 = track.bbox
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), track.color, 2)
            
            # Draw trail
            if viz_config['show_trail'] and len(track.history) > 1:
                for i in range(1, len(track.history)):
                    pt1 = (int(track.history[i-1][0]), int(track.history[i-1][1]))
                    pt2 = (int(track.history[i][0]), int(track.history[i][1]))
                    
                    # Fade the trail
                    alpha = i / len(track.history)
                    trail_color = tuple(int(c * alpha) for c in track.color)
                    cv2.line(annotated_frame, pt1, pt2, trail_color, 2)
            
            # Draw center point
            center = (int(track.centroid[0]), int(track.centroid[1]))
            cv2.circle(annotated_frame, center, 4, track.color, -1)
            
            # Prepare label
            label_parts = [track.class_name]
            if viz_config['show_track_id']:
                label_parts.append(f"ID:{track.track_id}")
            label_parts.append(f"{track.confidence:.2f}")
            
            label = " ".join(label_parts)
            
            # Draw label
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10), 
                         (x1 + label_width + 5, y1), track.color, -1)
            cv2.putText(annotated_frame, label, (x1 + 2, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame

    def process_video(self, video_path: str, output_path: str = None, show_live: bool = False):
        """Process video with object tracking"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n=== Week 3: Object Tracking ===")
        print(f"Video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_number = 0
        total_tracks_created = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect objects
                detections = self.detect_objects(frame)
                
                # Update tracks
                tracked_objects = self.update_tracks(detections)
                
                # Count new tracks
                current_max_id = max([t.track_id for t in tracked_objects], default=-1)
                if current_max_id >= total_tracks_created:
                    total_tracks_created = current_max_id + 1
                
                # Draw tracks
                annotated_frame = self.draw_tracks(frame, tracked_objects)
                
                # Add frame info
                info_lines = [
                    f"Week 3: Object Tracking",
                    f"Frame: {frame_number}/{total_frames}",
                    f"Active Tracks: {len(tracked_objects)}",
                    f"Total Tracks Created: {total_tracks_created}"
                ]
                
                for i, line in enumerate(info_lines):
                    cv2.putText(annotated_frame, line, (10, 25 + i * 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Show live preview
                if show_live:
                    cv2.imshow('Week 3: Object Tracking', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Write frame
                if writer:
                    writer.write(annotated_frame)
                
                frame_number += 1
                
                # Progress update
                if frame_number % 30 == 0:
                    progress = frame_number / total_frames * 100
                    print(f"Progress: {progress:.1f}% - Active tracks: {len(tracked_objects)}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_live:
                cv2.destroyAllWindows()
        
        print(f"Total tracks created: {total_tracks_created}")
        print(f"Average active tracks: {len(tracked_objects)}")


def main():
   
    parser = argparse.ArgumentParser(description='Week 3: Object Tracking')
    
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
        detector = TrackingDetector(args.model, args.config)
        
        # Process video
        detector.process_video(args.video, args.output, args.live)
        
        print("Week 3 challenge completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 