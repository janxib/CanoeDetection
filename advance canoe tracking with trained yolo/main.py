from ultralytics import YOLO
import cv2
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle

# Load model
model = YOLO("runs/detect/train4/weights/best.pt")

# Setup
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(3)), int(cap.get(4))

# Zoom out factor (0.7 means 70% of original size)
zoom_factor = 0.7
new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)

# Output video writer with zoomed out dimensions
out = cv2.VideoWriter("output_advanced.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (new_w, new_h))

# Tracking data
trajectory = []
unique_gates = {}  # Dictionary to store unique gates with their positions
gate_confidence = {}  # Track confidence scores for each gate
passed_gates = set()
gate_counter = 0
canoe_history = []  # Track recent canoe positions

frame_id = 0
past_points = []

def find_matching_gate(new_gate_center, existing_gates, threshold=150):
    """Find if a gate already exists within threshold distance"""
    new_x, new_y = new_gate_center
    best_match = None
    best_distance = float('inf')
    
    for gate_id, (existing_x, existing_y) in existing_gates.items():
        distance = ((new_x - existing_x)**2 + (new_y - existing_y)**2)**0.5
        if distance < threshold and distance < best_distance:
            best_match = gate_id
            best_distance = distance
    
    return best_match

def check_gate_passage(canoe_pos, gate_bounds, history_length=10):
    """Check if canoe has passed through a gate using trajectory history"""
    if len(canoe_history) < 2:
        return False
    
    gx1, gy1, gx2, gy2 = gate_bounds
    gate_center_x = (gx1 + gx2) / 2
    gate_center_y = (gy1 + gy2) / 2
    
    # Check multiple conditions for gate passage
    for i in range(max(0, len(canoe_history) - history_length), len(canoe_history)):
        cx, cy = canoe_history[i]
        
        # Method 1: Check if canoe is within gate bounds with generous buffer
        buffer = 40
        if ((gx1 - buffer) <= cx <= (gx2 + buffer) and 
            (gy1 - buffer) <= cy <= (gy2 + buffer)):
            return True
        
        # Method 2: Check distance to gate center
        distance_to_center = ((cx - gate_center_x)**2 + (cy - gate_center_y)**2)**0.5
        if distance_to_center < 50:  # Within 50 pixels of gate center
            return True
        
        # Method 3: Check if canoe overlaps with gate area
        gate_width = gx2 - gx1
        gate_height = gy2 - gy1
        overlap_buffer = min(gate_width, gate_height) * 0.3  # 30% of gate size
        if ((gx1 - overlap_buffer) <= cx <= (gx2 + overlap_buffer) and 
            (gy1 - overlap_buffer) <= cy <= (gy2 + overlap_buffer)):
            return True
    
    return False

def create_tracking_graph():
    """Create comprehensive tracking visualization"""
    plt.style.use('dark_background')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Canoe Slalom Tracking Analysis', fontsize=16, fontweight='bold')
    
    # Extract trajectory data
    if trajectory:
        x_coords = [point['x'] for point in trajectory]
        y_coords = [point['y'] for point in trajectory]
        frames = [point['frame'] for point in trajectory]
    else:
        x_coords, y_coords, frames = [], [], []
    
    # Plot 1: Full trajectory with gates
    ax1.set_title('Canoe Trajectory & Gate Positions', fontsize=14, fontweight='bold')
    
    if x_coords and y_coords:
        # Plot trajectory path
        ax1.plot(x_coords, y_coords, 'cyan', linewidth=2, alpha=0.7, label='Canoe Path')
        ax1.scatter(x_coords[0], y_coords[0], color='lime', s=100, marker='o', label='Start', zorder=5)
        ax1.scatter(x_coords[-1], y_coords[-1], color='red', s=100, marker='s', label='End', zorder=5)
        
        # Plot trajectory direction arrows
        for i in range(0, len(x_coords)-1, max(1, len(x_coords)//20)):
            dx = x_coords[i+1] - x_coords[i]
            dy = y_coords[i+1] - y_coords[i]
            ax1.arrow(x_coords[i], y_coords[i], dx*0.3, dy*0.3, 
                     head_width=10, head_length=15, fc='yellow', ec='yellow', alpha=0.6)
    
    # Plot gates
    for gate_id, (gate_x, gate_y) in unique_gates.items():
        if gate_id in passed_gates:
            ax1.scatter(gate_x, gate_y, color='lime', s=150, marker='s', alpha=0.8, label='Passed Gate' if gate_id == list(passed_gates)[0] else "")
            ax1.annotate(gate_id.split('_')[1], (gate_x, gate_y), xytext=(5, 5), 
                        textcoords='offset points', color='lime', fontsize=8, fontweight='bold')
        else:
            ax1.scatter(gate_x, gate_y, color='red', s=150, marker='s', alpha=0.8, label='Missed Gate' if gate_id == list(set(unique_gates.keys()) - passed_gates)[0] else "")
            ax1.annotate(gate_id.split('_')[1], (gate_x, gate_y), xytext=(5, 5), 
                        textcoords='offset points', color='red', fontsize=8, fontweight='bold')
    
    ax1.set_xlabel('X Position (pixels)')
    ax1.set_ylabel('Y Position (pixels)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # Invert Y-axis to match video coordinates
    
    # Plot 2: Speed analysis
    ax2.set_title('Canoe Speed Analysis', fontsize=14, fontweight='bold')
    
    if len(x_coords) > 1:
        speeds = []
        for i in range(1, len(x_coords)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            speed = np.sqrt(dx**2 + dy**2)
            speeds.append(speed)
        
        ax2.plot(frames[1:], speeds, 'orange', linewidth=2, alpha=0.8)
        ax2.fill_between(frames[1:], speeds, alpha=0.3, color='orange')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Speed (pixels/frame)')
        ax2.grid(True, alpha=0.3)
        
        # Add speed statistics
        avg_speed = np.mean(speeds)
        max_speed = np.max(speeds)
        ax2.axhline(y=avg_speed, color='lime', linestyle='--', alpha=0.7, label=f'Avg: {avg_speed:.1f}')
        ax2.axhline(y=max_speed, color='red', linestyle='--', alpha=0.7, label=f'Max: {max_speed:.1f}')
        ax2.legend()
    
    # Plot 3: Gate performance
    ax3.set_title('Gate Performance Analysis', fontsize=14, fontweight='bold')
    
    total_gates = len(unique_gates)
    passed_count = len(passed_gates)
    missed_count = total_gates - passed_count
    
    labels = ['Passed', 'Missed']
    sizes = [passed_count, missed_count]
    colors = ['lime', 'red']
    explode = (0.05, 0.05)
    
    if total_gates > 0:
        wedges, texts, autotexts = ax3.pie(sizes, explode=explode, labels=labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
        
        # Add statistics text
        stats_text = f"""
        Total Gates: {total_gates}
        Passed: {passed_count}
        Missed: {missed_count}
        Success Rate: {passed_count/total_gates*100:.1f}%
        """
        ax3.text(1.3, 0.5, stats_text, transform=ax3.transAxes, fontsize=10, 
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="gray", alpha=0.3))
    
    # Plot 4: Gate distribution
    ax4.set_title('Gate Distribution & Trajectory Heatmap', fontsize=14, fontweight='bold')
    
    if x_coords and y_coords:
        # Create heatmap of trajectory
        heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax4.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', alpha=0.6)
        
        # Plot gates on heatmap
        for gate_id, (gate_x, gate_y) in unique_gates.items():
            if gate_id in passed_gates:
                ax4.scatter(gate_x, gate_y, color='lime', s=100, marker='s', alpha=0.9, edgecolors='black', linewidth=2)
            else:
                ax4.scatter(gate_x, gate_y, color='red', s=100, marker='s', alpha=0.9, edgecolors='black', linewidth=2)
        
        ax4.set_xlabel('X Position (pixels)')
        ax4.set_ylabel('Y Position (pixels)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
        cbar.set_label('Trajectory Density', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig('canoe_tracking_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Tracking analysis graph saved as: canoe_tracking_analysis.png")

def create_scenario_graphs():
    """Create scenario-specific analysis graphs"""
    
    # Graph 1: Gate Success Timeline
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Canoe Slalom Performance Analysis by Scenario', fontsize=16, fontweight='bold')
    
    # Scenario 1: Gate Success Timeline
    ax1.set_title('Scenario 1: Gate Success Timeline', fontsize=14, fontweight='bold')
    
    if unique_gates:
        gate_numbers = [int(gate_id.split('_')[1]) for gate_id in unique_gates.keys()]
        gate_numbers.sort()
        
        passed_numbers = [int(gate_id.split('_')[1]) for gate_id in passed_gates]
        missed_numbers = [int(gate_id.split('_')[1]) for gate_id in unique_gates.keys() if gate_id not in passed_gates]
        
        # Create timeline
        ax1.scatter(passed_numbers, [1]*len(passed_numbers), color='lime', s=100, marker='o', 
                   label=f'Passed ({len(passed_numbers)})', alpha=0.8, zorder=5)
        ax1.scatter(missed_numbers, [1]*len(missed_numbers), color='red', s=100, marker='x', 
                   label=f'Missed ({len(missed_numbers)})', alpha=0.8, zorder=5)
        
        # Add gate numbers as text
        for num in gate_numbers:
            color = 'lime' if num in passed_numbers else 'red'
            ax1.annotate(str(num), (num, 1), xytext=(0, 10), textcoords='offset points', 
                        ha='center', color=color, fontsize=10, fontweight='bold')
        
        ax1.set_xlabel('Gate Number')
        ax1.set_ylabel('Status')
        ax1.set_ylim(0.5, 1.5)
        ax1.set_xlim(0, max(gate_numbers) + 1)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_yticks([1])
        ax1.set_yticklabels(['Gates'])
    
    # Scenario 2: Performance vs Expected Path
    ax2.set_title('Scenario 2: Performance Analysis - Expected vs Actual', fontsize=14, fontweight='bold')
    
    if trajectory:
        x_coords = [point['x'] for point in trajectory]
        y_coords = [point['y'] for point in trajectory]
        
        # Calculate performance metrics
        total_distance = 0
        for i in range(1, len(x_coords)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        # Create performance comparison
        metrics = ['Gates Passed', 'Gates Missed', 'Total Distance\n(x100 px)', 'Avg Speed\n(px/frame)']
        actual_values = [len(passed_gates), len(unique_gates) - len(passed_gates), 
                        total_distance/100, np.mean([np.sqrt((x_coords[i] - x_coords[i-1])**2 + 
                                                           (y_coords[i] - y_coords[i-1])**2) 
                                                   for i in range(1, len(x_coords))]) if len(x_coords) > 1 else 0]
        
        # Theoretical ideal values (for comparison)
        expected_values = [len(unique_gates), 0, total_distance/120, actual_values[3]*0.8]  # Idealized performance
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax2.bar(x_pos - width/2, actual_values, width, label='Actual', color='orange', alpha=0.7)
        bars2 = ax2.bar(x_pos + width/2, expected_values, width, label='Expected/Ideal', color='lime', alpha=0.7)
        
        ax2.set_xlabel('Performance Metrics')
        ax2.set_ylabel('Values')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('canoe_scenario_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Scenario analysis graph saved as: canoe_scenario_analysis.png")

def create_simple_trajectory_plot():
    """Create a clean, simple trajectory plot for presentations"""
    plt.figure(figsize=(12, 8))
    plt.style.use('default')  # Use default style for clean look
    
    if trajectory:
        x_coords = [point['x'] for point in trajectory]
        y_coords = [point['y'] for point in trajectory]
        
        # Plot trajectory
        plt.plot(x_coords, y_coords, 'b-', linewidth=3, alpha=0.7, label='Canoe Path')
        
        # Mark start and end
        plt.scatter(x_coords[0], y_coords[0], color='green', s=200, marker='o', 
                   label='Start', zorder=5, edgecolors='black', linewidth=2)
        plt.scatter(x_coords[-1], y_coords[-1], color='red', s=200, marker='s', 
                   label='End', zorder=5, edgecolors='black', linewidth=2)
        
        # Plot gates
        for gate_id, (gate_x, gate_y) in unique_gates.items():
            gate_num = gate_id.split('_')[1]
            if gate_id in passed_gates:
                plt.scatter(gate_x, gate_y, color='lime', s=150, marker='s', 
                           alpha=0.8, edgecolors='black', linewidth=1, zorder=4)
                plt.annotate(gate_num, (gate_x, gate_y), xytext=(0, 20), 
                            textcoords='offset points', ha='center', 
                            fontsize=10, fontweight='bold', color='green')
            else:
                plt.scatter(gate_x, gate_y, color='red', s=150, marker='s', 
                           alpha=0.8, edgecolors='black', linewidth=1, zorder=4)
                plt.annotate(gate_num, (gate_x, gate_y), xytext=(0, 20), 
                            textcoords='offset points', ha='center', 
                            fontsize=10, fontweight='bold', color='red')
        
        # Add direction arrows
        arrow_spacing = max(1, len(x_coords) // 15)
        for i in range(0, len(x_coords)-1, arrow_spacing):
            dx = x_coords[i+1] - x_coords[i]
            dy = y_coords[i+1] - y_coords[i]
            plt.arrow(x_coords[i], y_coords[i], dx*0.5, dy*0.5, 
                     head_width=15, head_length=20, fc='blue', ec='blue', alpha=0.5)
        
        plt.title('Canoe Slalom - Trajectory and Gate Performance', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('X Position (pixels)', fontsize=12)
        plt.ylabel('Y Position (pixels)', fontsize=12)
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # Invert Y-axis to match video coordinates
        
        # Add performance text
        performance_text = f"""
        Performance Summary:
        • Total Gates: {len(unique_gates)}
        • Passed: {len(passed_gates)} ({len(passed_gates)/len(unique_gates)*100:.1f}%)
        • Missed: {len(unique_gates) - len(passed_gates)} ({(len(unique_gates) - len(passed_gates))/len(unique_gates)*100:.1f}%)
        """
        plt.text(0.02, 0.98, performance_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                fontsize=11)
    
    plt.tight_layout()
    plt.savefig('canoe_trajectory_simple.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Simple trajectory plot saved as: canoe_trajectory_simple.png")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    boxes = results[0].boxes.data
    annotated = results[0].plot()

    canoe_center = None
    current_frame_gates = []
    best_canoe = None
    best_canoe_conf = 0

    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf, cls_id = box.tolist()
        cls_id = int(cls_id)
        label = model.names[cls_id]

        if label == "canoes":
            # Only keep the canoe with highest confidence per frame
            if conf > best_canoe_conf:
                xc, yc = int((x1 + x2) / 2), int((y1 + y2) / 2)
                best_canoe = (xc, yc)
                best_canoe_conf = conf

        elif label == "gate":
            # More aggressive gate detection - lower thresholds but smarter tracking
            gate_width = x2 - x1
            gate_height = y2 - y1
            gate_area = gate_width * gate_height
            
            # More lenient filtering to catch all real gates
            if (conf > 0.4 and  # Lower confidence threshold
                gate_width > 25 and gate_height > 25 and  # Smaller minimum size
                gate_area > 800 and gate_area < 100000):  # Wider area range
                
                # Calculate gate center
                gate_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                
                # Check if this gate already exists
                existing_gate_id = find_matching_gate(gate_center, unique_gates)
                
                if existing_gate_id is None:
                    # Create new gate more aggressively
                    if gate_counter < 35:  # Higher limit with safety margin
                        gate_counter += 1
                        gate_id = f"gate_{gate_counter}"
                        unique_gates[gate_id] = gate_center
                        gate_confidence[gate_id] = conf
                        print(f"New gate created: {gate_id} at frame {frame_id} (conf: {conf:.2f})")
                    else:
                        # Use existing gate with lowest confidence if we're over limit
                        if gate_confidence:
                            worst_gate = min(gate_confidence, key=gate_confidence.get)
                            if conf > gate_confidence[worst_gate]:
                                gate_id = worst_gate
                                unique_gates[gate_id] = gate_center
                                gate_confidence[gate_id] = conf
                            else:
                                continue
                        else:
                            continue
                else:
                    # Existing gate - update confidence if better
                    gate_id = existing_gate_id
                    if gate_id in gate_confidence:
                        gate_confidence[gate_id] = max(gate_confidence[gate_id], conf)
                    else:
                        gate_confidence[gate_id] = conf
                
                current_frame_gates.append((gate_id, (int(x1), int(y1), int(x2), int(y2))))

    # Set the best canoe as the tracked canoe for this frame
    if best_canoe is not None:
        canoe_center = best_canoe
        trajectory.append({"frame": frame_id, "x": canoe_center[0], "y": canoe_center[1]})
        
        # Maintain canoe history for trajectory-based gate detection
        canoe_history.append(canoe_center)
        if len(canoe_history) > 20:  # Keep last 20 positions
            canoe_history.pop(0)

    # Ultra-aggressive gate passing detection using multiple methods
    if canoe_center:
        past_points.append(canoe_center)
        
        # Check all gates in the current frame AND all previously detected gates
        all_gates_to_check = current_frame_gates.copy()
        
        # Also check all previously detected gates within reasonable distance
        for gate_id, gate_center in unique_gates.items():
            if gate_id not in [gid for gid, _ in current_frame_gates]:
                # Estimate gate bounds based on average gate size (assuming ~60x60 pixels)
                estimated_bounds = (gate_center[0] - 30, gate_center[1] - 30, 
                                  gate_center[0] + 30, gate_center[1] + 30)
                all_gates_to_check.append((gate_id, estimated_bounds))
        
        for gate_id, (gx1, gy1, gx2, gy2) in all_gates_to_check:
            # Skip if already passed
            if gate_id in passed_gates:
                continue
                
            # Multiple detection methods for maximum accuracy
            passed = False
            
            # Method 1: Direct overlap detection (most sensitive)
            gate_center_x = (gx1 + gx2) / 2
            gate_center_y = (gy1 + gy2) / 2
            distance_to_gate = ((canoe_center[0] - gate_center_x)**2 + (canoe_center[1] - gate_center_y)**2)**0.5
            
            if distance_to_gate < 60:  # Very close to gate center
                passed = True
                print(f"✅ Gate {gate_id} PASSED - Direct overlap (distance: {distance_to_gate:.1f})")
            
            # Method 2: Current position with generous buffer
            if not passed:
                buffer = 35
                if (gx1 - buffer) <= canoe_center[0] <= (gx2 + buffer) and (gy1 - buffer) <= canoe_center[1] <= (gy2 + buffer):
                    passed = True
                    print(f"✅ Gate {gate_id} PASSED - Position buffer")
            
            # Method 3: Trajectory-based detection
            if not passed:
                passed = check_gate_passage(canoe_center, (gx1, gy1, gx2, gy2))
                if passed:
                    print(f"✅ Gate {gate_id} PASSED - Trajectory analysis")
            
            # Method 4: Check recent canoe positions for any overlap
            if not passed and len(canoe_history) >= 3:
                for i in range(max(0, len(canoe_history) - 5), len(canoe_history)):
                    cx, cy = canoe_history[i]
                    dist = ((cx - gate_center_x)**2 + (cy - gate_center_y)**2)**0.5
                    if dist < 45:  # Within 45 pixels in recent history
                        passed = True
                        print(f"✅ Gate {gate_id} PASSED - Recent history (distance: {dist:.1f})")
                        break
            
            # Method 5: Bounding box intersection
            if not passed:
                canoe_size = 20  # Assume canoe is ~20x20 pixels
                canoe_bounds = (canoe_center[0] - canoe_size, canoe_center[1] - canoe_size,
                               canoe_center[0] + canoe_size, canoe_center[1] + canoe_size)
                
                # Check if bounding boxes intersect
                if (canoe_bounds[0] <= gx2 and canoe_bounds[2] >= gx1 and
                    canoe_bounds[1] <= gy2 and canoe_bounds[3] >= gy1):
                    passed = True
                    print(f"✅ Gate {gate_id} PASSED - Bounding box intersection")
            
            if passed:
                passed_gates.add(gate_id)

    # Add gate labels and information to the frame
    for gate_id, (gx1, gy1, gx2, gy2) in current_frame_gates:
        # Gate center for label positioning
        gate_center_x = int((gx1 + gx2) / 2)
        gate_center_y = int((gy1 + gy2) / 2)
        
        # Extract gate number from gate_id (e.g., "gate_1" -> "1")
        gate_number = gate_id.split('_')[1]
        
        # Calculate distance to canoe if canoe is detected
        distance_to_canoe = ""
        if canoe_center:
            dist = ((canoe_center[0] - gate_center_x)**2 + (canoe_center[1] - gate_center_y)**2)**0.5
            distance_to_canoe = f" ({dist:.0f}px)"
        
        # Determine if gate has been passed
        if gate_id in passed_gates:
            color = (0, 255, 0)  # Green for passed gates
            status = "PASSED"
        else:
            color = (0, 0, 255)  # Red for not passed gates
            status = "ACTIVE"
        
        # Draw gate number with distance
        cv2.putText(annotated, f"Gate {gate_number}{distance_to_canoe}", 
                   (gate_center_x - 40, gate_center_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw status
        cv2.putText(annotated, status, 
                   (gate_center_x - 25, gate_center_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw a circle at gate center - larger if canoe is very close
        circle_size = 5 if canoe_center and dist < 60 else 3
        cv2.circle(annotated, (gate_center_x, gate_center_y), circle_size, color, -1)
        
        # Draw connection line if canoe is very close
        if canoe_center and dist < 80:
            cv2.line(annotated, (gate_center_x, gate_center_y), canoe_center, (255, 255, 0), 2)

    # Highlight canoe position with a bright circle
    if canoe_center:
        cv2.circle(annotated, canoe_center, 8, (0, 255, 255), 3)  # Yellow circle around canoe
        cv2.putText(annotated, "CANOE", (canoe_center[0] - 20, canoe_center[1] - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Add overall statistics to the frame
    stats_y = 30
    cv2.putText(annotated, f"Total Gates: {len(unique_gates)}", 
               (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, f"Passed: {len(passed_gates)}", 
               (10, stats_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated, f"Missed: {len(unique_gates) - len(passed_gates)}", 
               (10, stats_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(annotated, f"Frame: {frame_id}", 
               (10, stats_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show current canoe position
    if canoe_center:
        cv2.putText(annotated, f"Canoe: ({canoe_center[0]}, {canoe_center[1]})", 
                   (10, stats_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Zoom out the frame
    zoomed_frame = cv2.resize(annotated, (new_w, new_h))
    
    # Show and save frame
    out.write(zoomed_frame)
    cv2.imshow("Advanced Tracking", zoomed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# Post-processing: Filter out gates with very low confidence if we have too many
if len(unique_gates) > 30:
    # Sort gates by confidence and keep top 27-30
    sorted_gates = sorted(gate_confidence.items(), key=lambda x: x[1], reverse=True)
    top_gates = dict(sorted_gates[:30])
    
    # Update tracking data to only include top gates
    filtered_unique_gates = {k: v for k, v in unique_gates.items() if k in top_gates}
    filtered_passed_gates = {g for g in passed_gates if g in top_gates}
    
    print(f"🔧 Post-processing: Filtered from {len(unique_gates)} to {len(filtered_unique_gates)} gates")
    unique_gates = filtered_unique_gates
    passed_gates = filtered_passed_gates

# Save JSON (after post-processing)
tracking_output = {
    "trajectory": trajectory,
    "gates_detected": list(unique_gates.keys()),
    "gates_passed": list(passed_gates),
    "gates_missed": list(set(unique_gates.keys()) - passed_gates),
    "gate_confidence_scores": gate_confidence
}

with open("canoe_tracking_data.json", "w") as f:
    json.dump(tracking_output, f, indent=2)

print("✅ Tracking complete.")
print("🎥 Video saved as: output_advanced.mp4")
print("📄 JSON saved as: canoe_tracking_data.json")
print(f"🎯 Gates detected: {len(unique_gates)}")
print(f"✅ Gates passed: {len(passed_gates)}")
print(f"❌ Gates missed: {len(unique_gates) - len(passed_gates)}")
print(f"📊 Gate detection efficiency: {len(unique_gates)}/27 = {len(unique_gates)/27*100:.1f}%")
print(f"🎯 Pass rate: {len(passed_gates)}/{len(unique_gates)} = {len(passed_gates)/len(unique_gates)*100:.1f}%")

# Generate comprehensive tracking analysis graph
print("\n🔄 Generating tracking analysis graph...")
create_tracking_graph()

# Generate scenario-specific analysis graphs
print("\n🔄 Generating scenario analysis graphs...")
create_scenario_graphs()

# Generate simple trajectory plot
print("\n🔄 Generating simple trajectory plot...")
create_simple_trajectory_plot()
