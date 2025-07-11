# Week 1: Basic YOLO Object Detection

## Learning Objectives
- Understand YOLO model basics
- Learn to load and run object detection
- Basic visualization of detection results
- Process video files frame by frame

## What You'll Build
A simple object detector that can identify various objects in video frames using YOLO.

## Key Concepts
- **YOLO Model**: You Only Look Once - real-time object detection
- **Bounding Boxes**: Rectangular boxes around detected objects
- **Confidence Scores**: How certain the model is about a detection
- **Class IDs**: Numerical identifiers for object types

## Files
- `basic_yolo_detector.py` - Main detection script
- `yolov8n.pt` - YOLO model weights
- `README.md` - This file

## Command to Run

```bash
python basic_yolo_detector.py --video "../data/videos/test.mp4" --output "week1_output.mp4" --live
```

## Expected Output
- Detects all objects (people, boats, cars, etc.)
- Shows bounding boxes with class names and confidence scores
- Color coding: Green for boats, Blue for people, Red for others
- Frame counter and detection count display

## Challenge Tasks
1. Run the basic detector on the test video
2. Experiment with different confidence thresholds
3. Observe what objects are detected
4. Note any false positives or missed detections

## Next Week Preview
In Week 2, you'll learn to filter specific object types and use configuration files to focus on water-related objects. 