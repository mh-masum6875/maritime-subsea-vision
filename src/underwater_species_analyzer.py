from ultralytics import YOLOE
import cv2
import numpy as np
import os
import time
from collections import defaultdict

# Load YOLO model
model = YOLOE("yoloe-11l-seg.pt")
  
# Define specific underwater objects to detect
underwater_objects = [
    "fish", "coral reef", "aquatic plant", "trash", "sea anemone"
]

# Set classes for the model
model.set_classes(underwater_objects, model.get_text_pe(underwater_objects))

print(f"YOLOE model configured to detect {len(underwater_objects)} underwater object types")
print(f"Objects: {underwater_objects}")

# Video paths
video_folder = r"./data"
video_files = ["143039.mp4", "144000.mp4"]

# Output directory
output_dir = "runs/detect/underwater_objects_detection"
os.makedirs(output_dir, exist_ok=True)

def get_object_color(object_name):
    """Get color for object based on category"""
    name_lower = object_name.lower()
    
    # Fish - Green tones
    if "fish" in name_lower:
        return (0, 255, 0)  # Green
    
    # Marine life - Blue/Purple tones  
    elif any(keyword in name_lower for keyword in ["reef", "anemone", "aquatic", "plant"]):
        return (255, 127, 80)  # Orange-blue
    
    # Debris/objects - Red tones
    elif "trash" in name_lower:
        return (255, 0, 0)  # Red
    
    # Default color
    else:
        return (255, 255, 255)  # White

def draw_object_counter(frame, aggregate_detections, object_list):
    """Draw aggregate object counter on left side of frame with individual black backgrounds"""
    y_offset = 120  # Increased from 50 to add space from top
    line_height = 100  # Increased for much larger text
    font_scale = 2.4   # 2 times larger than before (was 1.2)
    font_thickness = 5  # Increased thickness
    x_position = 15
    padding = 10 # Increased padding for better text background border
    
    # Add title with black background
    title = "Total Detected:"
    (title_width, title_height), baseline = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
    # Draw black background for title with increased padding
    cv2.rectangle(frame, (x_position - padding, y_offset - title_height - padding), 
                  (x_position + title_width + padding, y_offset + padding), (0, 0, 0), -1)
    
    # Draw title text in white
    cv2.putText(frame, title, (x_position, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    y_offset += line_height
    
    # Draw object counts with individual black backgrounds
    for obj_name in object_list:
        count = aggregate_detections.get(obj_name, 0)
        
        # Capitalize first letter of each word
        capitalized_name = ' '.join(word.capitalize() for word in obj_name.split())
        text = f"{capitalized_name}: {count}"
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        # Draw black background for this line with increased padding
        cv2.rectangle(frame, (x_position - padding, y_offset - text_height - padding), 
                      (x_position + text_width + padding, y_offset + padding), (0, 0, 0), -1)
        
        # Draw text in white (no color coding)
        cv2.putText(frame, text, (x_position, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        y_offset += line_height

# Global detection counters
total_detections = defaultdict(int)
video_detections = {}

def process_video(video_path, video_name):
    """Process a single video and return detection counts"""
    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return {}
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
    
    # Setup output video
    output_path = os.path.join(output_dir, f"{video_name}_detected.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Detection counters for this video
    video_detection_count = defaultdict(int)
    frame_index = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        

        
        # Run detection with very low confidence to catch everything
        results = model.predict(frame, verbose=False, conf=0.01, agnostic_nms=True)[0]
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy()
            
            # Draw bounding boxes and count detections
            for i, (box, class_id, conf) in enumerate(zip(boxes, class_ids, confidences)):
                if class_id < len(underwater_objects):
                    object_name = underwater_objects[class_id]
                    
                    # Apply different confidence thresholds
                    if object_name.lower() in ["coral reef"]:
                        min_conf = 0.05  # Threshold for coral reef
                    else:
                        min_conf = 0.1   # Threshold for other objects
                    
                    # Skip if confidence is below threshold for this object
                    if conf < min_conf:
                        continue
                    
                    # Count detection for total video
                    video_detection_count[object_name] += 1
                    
                    # Get color for this object
                    bbox_color = get_object_color(object_name)
                    
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
                    
                    # Create label (object name only, no confidence)
                    label = object_name
                    
                    # Get text size for background rectangle
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    
                    # Draw background rectangle for text
                    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), bbox_color, -1)
                    
                    # Draw text
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw aggregate object counter on left side
        draw_object_counter(frame, video_detection_count, underwater_objects)
        
        # Add frame info
        frame_info = f"Frame: {frame_index}/{total_frames} | Video: {video_name}"
        cv2.putText(frame, frame_info, (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        frame_index += 1
        
        # Progress update every 100 frames
        if frame_index % 100 == 0:
            progress = (frame_index / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_index}/{total_frames} frames)")
    
    cap.release()
    out.release()
    
    processing_time = time.time() - start_time
    print(f"\nVideo processing completed in {processing_time:.2f} seconds")
    print(f"Output saved to: {output_path}")
    
    return dict(video_detection_count)

def print_detection_summary(video_name, detections):
    """Print detection summary for a video"""
    print(f"\n🎯 DETECTION SUMMARY - {video_name}")
    print("-" * 50)
    
    if not detections:
        print("No objects detected")
        return
    
    # Sort detections by count (highest first)
    sorted_detections = sorted(detections.items(), key=lambda x: x[1], reverse=True)
    
    total_objects = sum(detections.values())
    
    print(f"📊 Total Objects Detected: {total_objects}")
    print("\nDetected Objects:")
    
    for obj_name, count in sorted_detections:
        if count > 0:
            print(f"   {obj_name}: {count}")
    
    # Group by categories
    fish_keywords = ["fish"]
    marine_keywords = ["reef", "anemone", "aquatic", "plant"]
    debris_keywords = ["trash"]
    
    fish_total = sum(count for obj, count in detections.items() if any(keyword in obj.lower() for keyword in fish_keywords))
    marine_total = sum(count for obj, count in detections.items() if any(keyword in obj.lower() for keyword in marine_keywords))
    debris_total = sum(count for obj, count in detections.items() if any(keyword in obj.lower() for keyword in debris_keywords))
    
    if fish_total > 0 or marine_total > 0 or debris_total > 0:
        print(f"\nCategory Breakdown:")
        if fish_total > 0:
            print(f"   🐟 Fish & Marine Animals: {fish_total}")
        if marine_total > 0:
            print(f"   🪸 Coral & Marine Life: {marine_total}")
        if debris_total > 0:
            print(f"   🗑️ Debris & Objects: {debris_total}")

def generate_csv_report(video_detections, total_detections, output_dir):
    """Generate CSV report with all detection results"""
    import csv
    
    csv_path = os.path.join(output_dir, "underwater_detection_results.csv")
    
    # Get all unique object types from all detections
    all_objects = set()
    for detections in video_detections.values():
        all_objects.update(detections.keys())
    for obj in total_detections.keys():
        all_objects.add(obj)
    
    # Sort objects for consistent ordering
    sorted_objects = sorted(list(all_objects))
    
    # Create header
    header = ["Video_Name", "Total_Objects"] + sorted_objects + ["Fish_Marine_Animals_Total", "Coral_Marine_Life_Total", "Debris_Objects_Total"]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        # Write data for each video
        for video_name, detections in video_detections.items():
            row_data = [video_name, sum(detections.values())]
            
            # Add individual object counts
            for obj in sorted_objects:
                row_data.append(detections.get(obj, 0))
            
            # Calculate category totals
            fish_keywords = ["fish"]
            marine_keywords = ["reef", "anemone", "aquatic", "plant"]
            debris_keywords = ["trash"]
            
            fish_total = sum(count for obj, count in detections.items() if any(keyword in obj.lower() for keyword in fish_keywords))
            marine_total = sum(count for obj, count in detections.items() if any(keyword in obj.lower() for keyword in marine_keywords))
            debris_total = sum(count for obj, count in detections.items() if any(keyword in obj.lower() for keyword in debris_keywords))
            
            row_data.extend([fish_total, marine_total, debris_total])
            writer.writerow(row_data)
        
        # Write combined totals row
        combined_row = ["COMBINED_TOTAL", sum(total_detections.values())]
        for obj in sorted_objects:
            combined_row.append(total_detections.get(obj, 0))
        
        # Calculate combined category totals (using updated keywords)
        fish_keywords = ["fish"]
        marine_keywords = ["reef", "anemone", "aquatic", "plant"]
        debris_keywords = ["trash"]
        
        fish_total = sum(count for obj, count in total_detections.items() if any(keyword in obj.lower() for keyword in fish_keywords))
        marine_total = sum(count for obj, count in total_detections.items() if any(keyword in obj.lower() for keyword in marine_keywords))
        debris_total = sum(count for obj, count in total_detections.items() if any(keyword in obj.lower() for keyword in debris_keywords))
        
        combined_row.extend([fish_total, marine_total, debris_total])
        writer.writerow(combined_row)
    
    print(f"\n📊 CSV Report saved to: {csv_path}")
    return csv_path

# Main processing
print("🌊 UNDERWATER OBJECTS DETECTION SYSTEM")
print("=" * 60)
print(f"Processing videos from: {video_folder}")
print(f"Confidence thresholds: Coral Reef = 0.05, Others = 0.1")
print(f"Object classes: {len(underwater_objects)} selected objects only")
print(f"Objects to detect: {underwater_objects}")

# Process each video
for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    
    if os.path.exists(video_path):
        detections = process_video(video_path, video_file)
        video_detections[video_file] = detections
        
        # Add to total counts
        for obj, count in detections.items():
            total_detections[obj] += count
        
        # Print individual video summary
        print_detection_summary(video_file, detections)
    else:
        print(f"Warning: Video file not found: {video_path}")

# Print overall summary
print(f"\n{'='*60}")
print("🎯 OVERALL DETECTION SUMMARY")
print("=" * 60)

print_detection_summary("COMBINED VIDEOS", total_detections)

# Generate CSV report
csv_file = generate_csv_report(video_detections, total_detections, output_dir)

print(f"\n📁 Processed videos saved to: {output_dir}")
print("🎉 Underwater objects detection completed!") 