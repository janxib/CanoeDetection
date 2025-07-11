# Week 4: Advanced Canoe Detection and Tracking

## Learning Objectives
- Implement sophisticated canoe-specific detection
- Use intelligent scoring systems
- Apply advanced geometric analysis
- Combine detection, filtering, and tracking

## What You'll Build
A comprehensive canoe detection and tracking system that intelligently identifies canoe-like objects using multiple scoring criteria.

## Key Concepts
- **Canoe Scoring**: Multi-factor scoring for canoe-like characteristics
- **Geometric Analysis**: Size, aspect ratio, and shape evaluation
- **Detection Types**: Classification of different detection categories
- **Advanced Filtering**: Multi-pass filtering with strict criteria
- **Intelligent Tracking**: Enhanced tracking with prediction

## Files
- `advanced_canoe_tracker.py` - Complete advanced tracker
- `week4_config.json` - Advanced configuration with canoe-specific parameters
- `yolov8n.pt` - YOLO model weights
- `README.md` - This file

## Command to Run

```bash
python advanced_canoe_tracker.py --video "../data/videos/test.mp4" --config "week4_config.json" --output "week4_canoe_tracking.mp4" --live
```

## Advanced Features

### Canoe Scoring System
- **Size Score**: Evaluates object size against typical canoe dimensions
- **Shape Score**: Analyzes aspect ratio for canoe-like elongation
- **Class Bonus**: Extra points for boat/surfboard detections
- **Orientation Bonus**: Preference for horizontal objects

### Geometric Constraints
- Strict size limits (80-800px width, 20-200px height)
- Area constraints (2000-80000 square pixels)
- Aspect ratio requirements (3:1 to 15:1)
- Canoe-specific proportions (4:1 to 12:1 preferred)

### Detection Classification
- **boat**: Direct boat detections
- **canoe_like_surfboard**: Surfboards that score high as canoes
- **surfboard**: Regular surfboard detections
- **person_rejected**: People filtered out (rare due to high threshold)

## Expected Output
- High-precision canoe detection with minimal false positives
- Color-coded confidence visualization (Green=high, Yellow=medium, Orange=low)
- Detailed scoring information for each detection
- Trajectory trails for tracked canoes
- Comprehensive detection statistics

## Configuration Parameters

### Detection Settings
- Very strict confidence thresholds
- Limited to boat and surfboard classes
- High-quality geometric filtering

### Canoe Scoring
- Minimum score of 0.6 required
- Optimized for canoe characteristics
- Bonus scoring for ideal proportions

### Tracking
- Enhanced distance-based matching
- Trajectory history maintenance
- Intelligent track lifecycle management

## Challenge Tasks
1. Run the advanced tracker and compare to previous weeks
2. Analyze the canoe scoring system effectiveness
3. Experiment with scoring thresholds
4. Test on different video content
5. Evaluate false positive reduction

## Key Improvements Over Week 3
- Canoe-specific intelligence
- Multi-factor scoring system
- Advanced geometric analysis
- Significantly reduced false positives
- Comprehensive detection classification

## Real-World Applications
- Canoe race monitoring
- Water sports analysis
- Safety monitoring in water bodies
- Recreational activity tracking
- Sports performance analysis

## Success Metrics
- **Precision**: Percentage of detections that are actual canoes
- **Recall**: Percentage of actual canoes detected
- **Track Consistency**: How well objects maintain IDs
- **False Positive Rate**: Unwanted detections per frame 