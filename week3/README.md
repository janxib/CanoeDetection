# Week 3: Object Tracking

## Learning Objectives
- Understand object tracking concepts
- Maintain object identities across frames
- Create movement trails and trajectories
- Learn distance-based track association

## What You'll Build
A tracking system that assigns unique IDs to detected objects and follows their movement through the video.

## Key Concepts
- **Object Tracking**: Following objects across multiple frames
- **Track IDs**: Unique identifiers for each object
- **Centroid Tracking**: Using object centers for movement prediction
- **Track Association**: Matching detections to existing tracks
- **Track Lifecycle**: Creation, update, and deletion of tracks

## Files
- `tracking_detector.py` - Main script with tracking capabilities
- `yolov8n.pt` - YOLO model weights
- `README.md` - This file

## Command to Run

```bash
python tracking_detector.py --video "../data/videos/test.mp4" --output "week3_output.mp4" --live
```

## How Tracking Works
1. **Detection**: Find objects in current frame
2. **Association**: Match detections to existing tracks using distance
3. **Update**: Update track positions and properties
4. **Creation**: Create new tracks for unmatched detections
5. **Deletion**: Remove tracks that haven't been updated recently

## Expected Output
- Each detected object gets a unique Track ID
- Movement trails show object paths
- Track statistics (active tracks, total created)
- Objects maintain IDs as they move through frames

## Key Features
- **Distance-based matching**: Objects are matched based on proximity
- **Track aging**: Tracks are removed if not updated for too long
- **Trail visualization**: Shows recent movement history
- **Track statistics**: Real-time tracking information

## Challenge Tasks
1. Run the tracking detector on the test video
2. Observe how objects maintain their IDs
3. Experiment with tracking parameters (max_distance, max_age)
4. Count how many unique objects are tracked
5. Note any ID switches or lost tracks

## Tracking Parameters
- **max_distance**: Maximum distance for track association (100 pixels)
- **max_age**: How long to keep tracks without updates (20 frames)
- **trail_length**: How many trail points to show (30 points)

## Key Improvements Over Week 2
- Object identity preservation across frames
- Movement trail visualization
- Track lifecycle management
- Better understanding of object behavior

## Next Week Preview
In Week 4, you'll build the advanced canoe-specific tracker with sophisticated scoring and geometric analysis for accurate canoe detection. 