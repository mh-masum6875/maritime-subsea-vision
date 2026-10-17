from ultralytics import YOLOE
import cv2
import numpy as np
import os

# Load YOLO model
model = YOLOE("yoloe-11l-seg.pt")

# Define fish and fishing net classes (using correct RAM tag list names)
fish_net_names = [
    "fish", "salmon", "trout", "tuna", "bass", "carp", "fishing net", 
    "fishing", "fisherman", "fishing boat", "net", "fishing pole"
]

# Set classes for the model
model.set_classes(fish_net_names, model.get_text_pe(fish_net_names))

# Update video path to use a fish video
video_path = "Fish and Fishing Nets.mp4"
cap = cv2.VideoCapture(video_path)

# Update output path
output_path = "runs/detect/Fish_and_Fishing_Nets.mp4"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

frame_index = 0

# Define colors for different fish and fishing equipment types (for bbox visualization)
fish_net_colors = {
    "fish": (0, 255, 0),              # Green
    "salmon": (255, 182, 193),        # Light Pink
    "trout": (0, 255, 255),           # Yellow  
    "tuna": (128, 0, 128),            # Purple
    "bass": (255, 165, 0),            # Orange
    "carp": (0, 128, 255),            # Light Blue
    "fishing net": (255, 0, 0),       # Blue
    "fishing": (0, 0, 255),           # Red
    "fisherman": (255, 255, 0),       # Cyan
    "fishing boat": (255, 0, 255),    # Magenta
    "net": (128, 255, 0),             # Lime Green
    "fishing pole": (255, 128, 128)   # Light Red
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection with agnostic-nms to prevent flipping
    results = model.predict(frame, verbose=False, conf=0.1, agnostic_nms=True)[0]
    
    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()

        # Draw bounding boxes for detected fish and fishing equipment
        for i, (box, class_id, conf) in enumerate(zip(boxes, class_ids, confidences)):
            if class_id < len(fish_net_names):
                item_name = fish_net_names[class_id]
                bbox_color = fish_net_colors.get(item_name, (255, 255, 255))
                
                x1, y1, x2, y2 = box.astype(int)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
                
                # Create label with item name only (no confidence)
                label = item_name
                
                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # For fishing net, place label at center of bbox; for others, place at top
                if item_name == "fishing net":
                    # Calculate center position
                    center_x = x1 + (x2 - x1) // 2 - text_width // 2
                    center_y = y1 + (y2 - y1) // 2 + text_height // 2
                    
                    # Draw background rectangle for text at center
                    cv2.rectangle(frame, (center_x, center_y - text_height - 5), (center_x + text_width, center_y + 5), bbox_color, -1)
                    
                    # Draw text at center
                    cv2.putText(frame, label, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    # Draw background rectangle for text at top
                    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), bbox_color, -1)
                    
                    # Draw text at top
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                print(f"Frame {frame_index}: Detected {item_name} with confidence {conf:.2f}")

    out.write(frame)
    frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete. Total frames: {frame_index}")
print("Fish and fishing net detection completed!") 