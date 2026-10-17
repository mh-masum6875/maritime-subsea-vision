from ultralytics import YOLOE
import cv2
import numpy as np
import os

# Load YOLO model
model = YOLOE("yoloe-11l-seg.pt")

# Define debris/trash classes only (excluding boats and legitimate objects)
debris_classes = [
    # Floating debris/containers
    "barrel", "bucket", "can", "bottle", "cup", "bag", "trash", 
    # Food waste that could be debris
    "apple", "banana", "orange", "lemon", "watermelon", "coconut",
    # Lost/floating items that could be debris  
    "shoe", "tire", "rope", "buoy", "bobfloat", "float",
    # Plastic waste
    "plastic", "bottle screw", "cap"
]

# Set classes for the model - only debris items
model.set_classes(debris_classes, model.get_text_pe(debris_classes))

# Update video path to use a debris video
video_path = "Debris.mp4"
cap = cv2.VideoCapture(video_path)

# Update output path
output_path = "runs/detect/Debris.mp4"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

frame_index = 0

# Using single color for all detections

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection with very low confidence to catch all detections
    results = model.predict(frame, verbose=False, conf=0.01, agnostic_nms=True)[0]
    
    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()

        # Draw bounding boxes for detected debris
        for i, (box, class_id, conf) in enumerate(zip(boxes, class_ids, confidences)):
            # Get class name from our debris classes list
            if class_id < len(debris_classes):
                class_name = debris_classes[class_id]
            else:
                class_name = f"debris_{class_id}"
            
            bbox_color = (0, 255, 0)  # Single green color for all detections
            
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
            
            # Create label with class name and confidence
            label = f"{class_name}: {conf:.2f}"
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background rectangle for text
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), bbox_color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            print(f"Frame {frame_index}: Detected {class_name} with confidence {conf:.3f}")

    out.write(frame)
    frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete. Total frames: {frame_index}")
print("Debris detection completed! Check the output video to see detected trash and debris items.")