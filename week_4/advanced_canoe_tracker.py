#!/usr/bin/env python3
"""
Week 4: Advanced Canoe Detection and Tracking
=============================================
Sophisticated canoe-specific detection with scoring and geometric analysis
"""

import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")


@dataclass
class CanoeDetection:
    """Advanced canoe detection data class"""
    frame_number: int
    timestamp: float
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    track_id: Optional[int] = None
    centroid: Optional[Tuple[float, float]] = None
    detection_type: str = "unknown"
    canoe_score: float = 0.0
    size_score: float = 0.0
    shape_score: float = 0.0


class AdvancedCanoeTracker:
    """Advanced YOLO-based canoe tracker with intelligent scoring"""
    
    def __init__(self, model_path: str = "yolov8n.pt", config_path: Optional[str] = None):
        """Initialize the advanced tracker"""
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")
        
        self.config = self._load_config(config_path)
        self.detections: List[CanoeDetection] = []
        self.model_path = model_path
        self.model = None
        self.class_names = []
        self._load_model()
        
        # Advanced tracking
        self.next_track_id = 0
        self.active_tracks = {}
        self.frame_count = 0
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load enhanced configuration"""
        default_config = {
            "detection": {
                "base_confidence": 0.3,
                "iou_threshold": 0.4,
                "max_detections": 50,
                "class_thresholds": {
                    8: 0.4,   # boat
                    37: 0.5,  # surfboard
                    0: 0.8,   # person (very restrictive)
                }
            },
            "geometry": {
                "min_width": 80,
                "max_width": 800,
                "min_height": 20,
                "max_height": 200,
                "min_area": 2000,
                "max_area": 80000,
                "min_aspect_ratio": 3.0,
                "max_aspect_ratio": 15.0,
                "preferred_ratio_min": 4.0,
                "preferred_ratio_max": 12.0
            },
            "canoe_scoring": {
                "min_canoe_score": 0.6,
                "boat_bonus": 0.8,
                "surfboard_bonus": 0.6,
                "horizontal_bonus": 0.4,
                "size_bonus_range": [3000, 40000],
                "aspect_bonus_range": [4.0, 10.0]
            },
            "tracking": {
                "max_distance": 150,
                "max_age": 20,
                "min_confidence": 0.4
            },
            "visualization": {
                "show_all_scores": True,
                "show_detection_type": True,
                "color_by_confidence": True,
                "trajectory_length": 30
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                self._merge_config(default_config, user_config)
        
        return default_config
    
    def _merge_config(self, base: Dict, update: Dict):
        """Recursively merge configuration dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            print(f"Loading advanced YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
            else:
                self.class_names = {
                    0: 'person', 8: 'boat', 37: 'surfboard'
                }
            
            print(f"Model loaded. Target classes: {list(self.config['detection']['class_thresholds'].keys())}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _calculate_canoe_score(self, bbox: Tuple[float, float, float, float], 
                              class_id: int, confidence: float) -> Tuple[float, float, float]:
        """Calculate comprehensive canoe scoring"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1.0
        
        # Size scoring
        scoring = self.config['canoe_scoring']
        size_min, size_max = scoring['size_bonus_range']
        if size_min <= area <= size_max:
            size_score = 1.0
        elif area < size_min:
            size_score = max(0.0, area / size_min)
        else:
            size_score = max(0.0, 1.0 - (area - size_max) / size_max)
        
        # Shape scoring (aspect ratio)
        aspect_min, aspect_max = scoring['aspect_bonus_range']
        if aspect_min <= aspect_ratio <= aspect_max:
            shape_score = 1.0
        elif aspect_ratio < aspect_min:
            shape_score = max(0.0, aspect_ratio / aspect_min)
        else:
            shape_score = max(0.0, 1.0 - (aspect_ratio - aspect_max) / (aspect_max * 2))
        
        # Overall canoe score
        canoe_score = 0.0
        
        # Size and shape components
        canoe_score += size_score * 0.3
        canoe_score += shape_score * 0.4
        
        # Orientation bonus (prefer horizontal)
        if width > height:
            canoe_score += scoring['horizontal_bonus'] * 0.2
        
        # Class-specific bonuses
        if class_id == 8:  # boat
            canoe_score += scoring['boat_bonus'] * 0.1
        elif class_id == 37:  # surfboard
            canoe_score += scoring['surfboard_bonus'] * 0.1
        
        return canoe_score, size_score, shape_score

    def detect_objects(self, frame: np.ndarray) -> List[CanoeDetection]:
        """Advanced canoe detection with scoring"""
        if self.model is None:
            return []
        
        self.frame_count += 1
        
        # Run YOLO detection
        results = self.model(frame, 
                           conf=self.config['detection']['base_confidence'],
                           iou=self.config['detection']['iou_threshold'],
                           max_det=self.config['detection']['max_detections'],
                           verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    # Check if this class is relevant
                    if cls_id not in self.config['detection']['class_thresholds']:
                        continue
                    
                    # Apply class-specific confidence threshold
                    min_conf = self.config['detection']['class_thresholds'][cls_id]
                    if conf < min_conf:
                        continue
                    
                    # Calculate canoe scores
                    canoe_score, size_score, shape_score = self._calculate_canoe_score(box, cls_id, conf)
                    
                    # Apply minimum canoe score filter
                    if canoe_score < self.config['canoe_scoring']['min_canoe_score']:
                        continue
                    
                    # Apply geometric constraints
                    if not self._passes_geometric_filter(box):
                        continue
                    
                    # Classify detection type
                    detection_type = self._classify_detection(cls_id, canoe_score, box)
                    
                    # Calculate centroid
                    x1, y1, x2, y2 = box
                    centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                    
                    detection = CanoeDetection(
                        frame_number=self.frame_count,
                        timestamp=self.frame_count / 30.0,  # Assume 30 FPS
                        bbox=(x1, y1, x2, y2),
                        confidence=float(conf),
                        class_id=cls_id,
                        centroid=centroid,
                        detection_type=detection_type,
                        canoe_score=canoe_score,
                        size_score=size_score,
                        shape_score=shape_score
                    )
                    
                    detections.append(detection)
        
        return detections

    def _passes_geometric_filter(self, bbox: Tuple[float, float, float, float]) -> bool:
        """Check if detection passes geometric constraints"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = max(width, height) / min(width, height)
        
        geometry = self.config['geometry']
        
        if not (geometry['min_width'] <= width <= geometry['max_width']):
            return False
        if not (geometry['min_height'] <= height <= geometry['max_height']):
            return False
        if not (geometry['min_area'] <= area <= geometry['max_area']):
            return False
        if not (geometry['min_aspect_ratio'] <= aspect_ratio <= geometry['max_aspect_ratio']):
            return False
        
        return True

    def _classify_detection(self, class_id: int, canoe_score: float, 
                           bbox: Tuple[float, float, float, float]) -> str:
        """Classify the type of detection"""
        if class_id == 8:
            return "boat"
        elif class_id == 37:
            if canoe_score > 0.7:
                return "canoe_like_surfboard"
            else:
                return "surfboard"
        elif class_id == 0:
            return "person_rejected"  # Should be very rare due to high threshold
        else:
            return "unknown"

    def update_tracking(self, detections: List[CanoeDetection]) -> List[CanoeDetection]:
        """Advanced tracking with prediction"""
        tracked_detections = []
        max_distance = self.config['tracking']['max_distance']
        
        # Simple distance-based tracking
        for detection in detections:
            best_track_id = None
            best_distance = float('inf')
            
            for track_id, track_info in self.active_tracks.items():
                last_centroid = track_info['last_centroid']
                distance = np.sqrt((detection.centroid[0] - last_centroid[0])**2 + 
                                 (detection.centroid[1] - last_centroid[1])**2)
                
                if distance < best_distance and distance < max_distance:
                    best_distance = distance
                    best_track_id = track_id
            
            # Assign or create track
            if best_track_id is not None:
                detection.track_id = best_track_id
                self.active_tracks[best_track_id].update({
                    'last_centroid': detection.centroid,
                    'age': 0,
                    'canoe_score': detection.canoe_score
                })
            else:
                detection.track_id = self.next_track_id
                self.active_tracks[self.next_track_id] = {
                    'last_centroid': detection.centroid,
                    'age': 0,
                    'canoe_score': detection.canoe_score,
                    'history': [detection.centroid]
                }
                self.next_track_id += 1
            
            # Update track history
            if detection.track_id in self.active_tracks:
                history = self.active_tracks[detection.track_id].get('history', [])
                history.append(detection.centroid)
                if len(history) > self.config['visualization']['trajectory_length']:
                    history.pop(0)
                self.active_tracks[detection.track_id]['history'] = history
            
            tracked_detections.append(detection)
        
        # Age and remove old tracks
        tracks_to_remove = []
        for track_id, track_info in self.active_tracks.items():
            track_info['age'] += 1
            if track_info['age'] > self.config['tracking']['max_age']:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]
        
        return tracked_detections

    def draw_detections(self, frame: np.ndarray, detections: List[CanoeDetection]) -> np.ndarray:
        """Advanced visualization with scoring information"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
            
            # Color based on canoe score
            if self.config['visualization']['color_by_confidence']:
                score = detection.canoe_score
                if score > 0.8:
                    color = (0, 255, 0)  # Green for high canoe score
                elif score > 0.6:
                    color = (0, 255, 255)  # Yellow for medium
                else:
                    color = (0, 165, 255)  # Orange for low
            else:
                color = (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw centroid
            if detection.centroid:
                cx, cy = [int(coord) for coord in detection.centroid]
                cv2.circle(annotated_frame, (cx, cy), 4, (255, 0, 255), -1)
            
            # Prepare detailed label
            label_parts = []
            if self.config['visualization']['show_detection_type']:
                label_parts.append(detection.detection_type.replace('_', ' ').title())
            
            if self.config['visualization']['show_all_scores']:
                label_parts.append(f"C:{detection.confidence:.2f}")
                label_parts.append(f"S:{detection.canoe_score:.2f}")
            
            if detection.track_id is not None:
                label_parts.append(f"ID:{detection.track_id}")
            
            label = " ".join(label_parts)
            
            # Draw label
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10), 
                         (x1 + label_width + 5, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1 + 2, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw trajectory
            if (detection.track_id is not None and 
                detection.track_id in self.active_tracks):
                history = self.active_tracks[detection.track_id].get('history', [])
                if len(history) > 1:
                    for i in range(1, len(history)):
                        pt1 = (int(history[i-1][0]), int(history[i-1][1]))
                        pt2 = (int(history[i][0]), int(history[i][1]))
                        alpha = i / len(history)
                        traj_color = tuple(int(c * alpha) for c in color)
                        cv2.line(annotated_frame, pt1, pt2, traj_color, 2)
        
        return annotated_frame

    def process_video(self, video_path: str, output_path: Optional[str] = None, show_live: bool = False):
        """Process video with advanced canoe detection"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n=== Week 4: Advanced Canoe Detection ===")
        print(f"Video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect canoes
                detections = self.detect_objects(frame)
                
                # Update tracking
                tracked_detections = self.update_tracking(detections)
                
                # Store detections
                self.detections.extend(tracked_detections)
                
                # Draw annotations
                annotated_frame = self.draw_detections(frame, tracked_detections)
                
                # Add frame info
                info_lines = [
                    f"Week 4: Advanced Canoe Detection",
                    f"Frame: {frame_number}/{total_frames}",
                    f"Canoe Detections: {len(tracked_detections)}",
                    f"Active Tracks: {len(self.active_tracks)}",
                    f"Total Detections: {len(self.detections)}"
                ]
                
                for i, line in enumerate(info_lines):
                    cv2.putText(annotated_frame, line, (10, 25 + i * 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Show live preview
                if show_live:
                    cv2.imshow('Week 4: Advanced Canoe Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Write frame
                if writer:
                    writer.write(annotated_frame)
                
                frame_number += 1
                
                # Progress update
                if frame_number % 30 == 0:
                    progress = frame_number / total_frames * 100
                    print(f"Progress: {progress:.1f}% - Canoe detections: {len(self.detections)}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_live:
                cv2.destroyAllWindows()
        
        # Final statistics
        print(f"\n=== Week 4 Complete ===")
        print(f"Total canoe detections: {len(self.detections)}")
        print(f"Average detections per frame: {len(self.detections)/frame_number:.2f}")
        
        # Detection type statistics
        type_counts = {}
        for det in self.detections:
            type_counts[det.detection_type] = type_counts.get(det.detection_type, 0) + 1
        
        print(f"Detection breakdown:")
        for det_type, count in sorted(type_counts.items()):
            print(f"  {det_type}: {count}")


def main():
    """Main function for Week 4"""
    parser = argparse.ArgumentParser(description='Week 4: Advanced Canoe Detection')
    
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
        # Initialize tracker
        tracker = AdvancedCanoeTracker(args.model, args.config)
        
        # Process video
        tracker.process_video(args.video, args.output, args.live)
        
        print("Week 4 challenge completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 