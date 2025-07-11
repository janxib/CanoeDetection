# Week 2: Class Filtering and Configuration

## Learning Objectives
- Learn to filter specific object classes
- Use configuration files for parameters
- Apply geometric constraints
- Focus on water-related objects

## What You'll Build
An enhanced detector that focuses on specific object types using configuration files and applies basic geometric filtering.

## Key Concepts
- **Class Filtering**: Only detect specific object types
- **Configuration Files**: JSON files to store detection parameters
- **Geometric Constraints**: Filter by size, aspect ratio, etc.
- **Class-Specific Thresholds**: Different confidence levels per object type

## Files
- `filtered_detector.py` - Main detection script with filtering
- `week2_config.json` - Configuration file for target classes
- `yolov8n.pt` - YOLO model weights
- `README.md` - This file

## Command to Run

```bash
python filtered_detector.py --video "../data/videos/test.mp4" --config "week2_config.json" --output "week2_output.mp4" --live
```

## Configuration Explained
The `week2_config.json` file contains:
- **target_classes**: Which objects to detect (boats, surfboards, people)
- **confidence thresholds**: Minimum confidence per class
- **visualization settings**: How to display results
- **geometric filters**: Size and aspect ratio constraints

## Expected Output
- Only shows boats, surfboards, and people (with high confidence)
- Enhanced labels with size information and aspect ratios
- Geometric filtering removes obviously wrong detections
- Better focus on water-related activities

## Challenge Tasks
1. Run the filtered detector with the provided config
2. Modify the config to be more/less restrictive
3. Experiment with different class combinations
4. Add new geometric constraints
5. Create your own configuration file

## Key Improvements Over Week 1
- Focused detection on relevant objects
- Configurable parameters
- Basic geometric filtering
- Enhanced visualization with size info

## Next Week Preview
In Week 3, you'll learn object tracking to maintain object identities across frames and create movement trails. 