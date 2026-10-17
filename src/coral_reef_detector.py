from ultralytics import YOLOE
import cv2
import numpy as np
import os

# Load YOLO model
model = YOLOE("yoloe-11l-seg.pt")

# Define coral classes
coral_names = [
    "coral", "coral reef"
]

# Set classes for the model
model.set_classes(coral_names, model.get_text_pe(coral_names))

# Update video path to use a coral video
video_path = "Coral.mp4"
cap = cv2.VideoCapture(video_path)

# Update output path
output_path = "runs/detect/Coral.mp4"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

frame_index = 0

# Define colors for different coral types (for bbox visualization)
coral_colors = {
    "coral": (0, 255, 255),        # Yellow
    "coral reef": (255, 0, 255)    # Magenta
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection with agnostic-nms to prevent flipping
    results = model.predict(frame, verbose=False, conf=0.25, agnostic_nms=True)[0]
    
    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()

        # Draw bounding boxes for detected coral
        for i, (box, class_id, conf) in enumerate(zip(boxes, class_ids, confidences)):
            if class_id < len(coral_names):
                coral_name = coral_names[class_id]
                bbox_color = coral_colors.get(coral_name, (255, 255, 255))
                
                x1, y1, x2, y2 = box.astype(int)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
                
                # Create label with coral name
                label = f"{coral_name}"
                
                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Draw background rectangle for text
                cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), bbox_color, -1)
                
                # Draw text
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                print(f"Frame {frame_index}: Detected {coral_name} with confidence {conf:.2f}")

    out.write(frame)
    frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete. Total frames: {frame_index}")
print("Coral detection completed!")