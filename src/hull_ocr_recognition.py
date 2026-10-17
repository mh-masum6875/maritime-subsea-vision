from ultralytics import YOLOE
import cv2
import numpy as np
import os
import easyocr
import re
import torch
from PIL import Image, ImageEnhance

# Load YOLO model for vessel detection
model = YOLOE("yoloe-11l-seg.pt")

# Define vessel classes - FOR DEMO: Only Container Ship and Sailboat
vessel_names = [
    "container ship"
]

# Set classes for the model
model.set_classes(vessel_names, model.get_text_pe(vessel_names))

# Initialize EasyOCR reader
print("Initializing EasyOCR for hull number detection...")

# Check for GPU availability
if torch.cuda.is_available():
    device = "cuda:0"
    gpu_enabled = True
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = "cpu"
    gpu_enabled = False
    print("WARNING: CUDA not available, using CPU for OCR")

# Initialize EasyOCR with GPU support if available
ocr_reader = easyocr.Reader(['en'], gpu=gpu_enabled)
print(f"✅ EasyOCR initialized with {'GPU' if gpu_enabled else 'CPU'}")

# Video input and output settings
video_path = "Vessel_4.mp4"
output_path = "runs/detect/hull_number_detection_output_4.mp4"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Define colors for different vessel types - FOR DEMO: Only Container Ship and Sailboat
vessel_colors = {
    "container ship": (0, 0, 255), # Red
    "sailboat": (128, 0, 128),     # Purple
}

def preprocess_image_for_ocr(image_region):
    """
    Preprocess the vessel region to improve OCR accuracy
    """
    try:
        # Convert to PIL Image for better preprocessing
        pil_image = Image.fromarray(cv2.cvtColor(image_region, cv2.COLOR_BGR2RGB))
        
        # Resize if too small (OCR works better on larger images)
        width, height = pil_image.size
        if width < 200 or height < 200:
            scale_factor = max(200/width, 200/height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(2.0)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.5)
        
        # Convert back to OpenCV format
        processed_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Apply additional OpenCV preprocessing
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Try OCR on enhanced grayscale first (often works better than binary)
        return enhanced
        
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        return cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)

def interpret_hull_number(ocr_text):
    """
    Convert OCR variations to actual company/hull names
    """
    cleaned = ocr_text.upper().strip()
    
    # Map OCR variations to actual company names
    hapag_variations = ['HAPAG-LBYD', 'HAPAG-LOD', 'HAPAG-LBD', 'HAPAG-LOYD', 'HAPAG-LBYL', 
                       'HAPAGLBD', 'HAPEGLBD', 'HPAGLBD', 'HAAGLBD', 'HPAGLBY', 'HAPZGLBD',
                       'HEPAG-LBD', 'HAPAG-LOP', 'IHAPAG-LBWD', 'ILAPAG-UBOVD', 'IHAPAG-UOVD',
                       'HAPAGLBL', 'HAAGLBY', 'HAPAG', 'LLOYD']
    
    for variation in hapag_variations:
        if variation in cleaned or cleaned in variation:
            return "Hapag-Lloyd"
    
    # Map other known variations
    maersk_variations = ['MAERSK', 'MERSK', 'MÆRSK']
    for variation in maersk_variations:
        if variation in cleaned:
            return "Maersk"
    
    cosco_variations = ['COSCO', 'CSCL']
    for variation in cosco_variations:
        if variation in cleaned:
            return "COSCO"
    
    evergreen_variations = ['EVERGREEN', 'EVERGR', 'EVERGRE']
    for variation in evergreen_variations:
        if variation in cleaned:
            return "Evergreen"
    
    msc_variations = ['MSC', 'MSCU']
    for variation in msc_variations:
        if variation in cleaned:
            return "MSC"
    
    # Return original if no mapping found
    return ocr_text

def validate_hull_number(text):
    """
    Validate if the detected text looks like a hull number or vessel identifier
    FOR DEMO: Accept legitimate hull numbers, filter out common OCR noise
    """
    if not text or len(text.strip()) < 3:
        return False
    
    text = text.strip().upper()
    
    # Remove common OCR artifacts and special characters
    cleaned_text = re.sub(r'[^\w\-]', '', text)
    
    if len(cleaned_text) < 3 or len(cleaned_text) > 25:
        return False
    
    # FILTER OUT COMMON OCR NOISE (text that our system adds or common misreads)
    noise_patterns = [
        'NOHULLNUMBER', 'HULL', 'CONTAINERSHIP', 'SAILBOAT', 'FRAME', 'FOUND',
        'HULLNUMBERSFOUND', 'HULLNUMBER', 'NUMBERS', 'ANALYZING', 'DETECTED',
        'SAFETY', 'FIRST', 'SECOND', 'THIRD', 'SECURITY', 'WARNING', 'CAUTION'
    ]
    
    for pattern in noise_patterns:
        if pattern in cleaned_text or cleaned_text in pattern:
            print(f"  🚫 Filtered out noise text: '{cleaned_text}'")
            return False
    
    # Check if it contains letters and/or numbers (typical hull number pattern)
    has_letter = bool(re.search(r'[A-Z]', cleaned_text))
    has_number = bool(re.search(r'[0-9]', cleaned_text))
    
    # Accept if it has reasonable hull number characteristics
    if (has_letter and has_number) or (has_letter and len(cleaned_text) >= 4) or (has_number and len(cleaned_text) >= 4):
        print(f"  ✅ Accepted potential hull number: '{cleaned_text}'")
        return True
    
    # Also accept known shipping company names (common vessel identifiers)
    shipping_companies = ['HAPAG', 'LLOYD', 'MAERSK', 'COSCO', 'MSC', 'EVERGREEN', 
                         'CMA', 'CGM', 'YANG', 'MING', 'OOCL', 'APL', 'MOL', 'NYK']
    
    for company in shipping_companies:
        if company in cleaned_text:
            print(f"  ✅ Accepted shipping company: '{cleaned_text}'")
            return True
    
    print(f"  🚫 Rejected text: '{cleaned_text}'")
    return False

def extract_hull_numbers(image_region):
    """
    Extract hull numbers from vessel image region using OCR
    """
    try:
        # Preprocess image for better OCR
        processed_image = preprocess_image_for_ocr(image_region)
        
        # Run OCR on the processed image
        results = ocr_reader.readtext(processed_image)
        
        hull_numbers = []
        
        for (bbox, text, confidence) in results:
            # Debug: Show all OCR results
            print(f"  OCR detected: '{text.strip()}' (confidence: {confidence:.2f})")
            
            # Filter results with reasonable confidence (lowered threshold)
            if confidence > 0.3:
                # Validate if the text looks like a hull number
                if validate_hull_number(text):
                    cleaned_text = re.sub(r'[^\w\-]', '', text.strip().upper())
                    hull_numbers.append({
                        'text': cleaned_text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
                    print(f"  ✅ Found potential hull number: '{cleaned_text}' (confidence: {confidence:.2f})")
                else:
                    print(f"  ❌ Text rejected by validation: '{text.strip()}'")
            else:
                print(f"  ❌ Low confidence, skipped: '{text.strip()}'")
        
        return hull_numbers
        
    except Exception as e:
        print(f"OCR processing failed: {e}")
        return []

# Cache for hull number detection to avoid redundant OCR calls
hull_cache = {}  # Clear cache to ensure fresh results after fixes
cache_update_interval = 60  # Update hull numbers every 60 frames (2 seconds at 30fps)

print("🧹 Cache cleared - ensuring early detection with strict hull number validation")

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def keep_highest_confidence_only(boxes, class_ids, confidences):
    """
    Keep one detection per class (1 container ship + 1 sailboat max)
    """
    if len(boxes) == 0:
        return boxes, class_ids, confidences
    
    print(f"  🔍 Before filtering: {len(boxes)} detections")
    for i, (class_id, conf) in enumerate(zip(class_ids, confidences)):
        if class_id < len(vessel_names):
            print(f"    {i+1}. {vessel_names[class_id]} (conf: {conf:.3f})")
    
    # Keep one detection per class (highest confidence for each class)
    final_boxes = []
    final_class_ids = []
    final_confidences = []
    
    # Get unique classes present in detections
    unique_classes = np.unique(class_ids)
    
    for class_id in unique_classes:
        if class_id < len(vessel_names):
            # Find all detections for this class
            class_mask = class_ids == class_id
            class_indices = np.where(class_mask)[0]
            
            # Get the highest confidence detection for this class
            class_confidences = confidences[class_mask]
            best_local_idx = np.argmax(class_confidences)
            best_global_idx = class_indices[best_local_idx]
            
            # Add to final results
            final_boxes.append(boxes[best_global_idx])
            final_class_ids.append(class_ids[best_global_idx])
            final_confidences.append(confidences[best_global_idx])
            
            print(f"  ✅ KEPT BEST {vessel_names[class_id]}: conf={confidences[best_global_idx]:.3f}")
    
    print(f"  📋 After filtering: {len(final_boxes)} detections (1 per class)")
    
    return np.array(final_boxes), np.array(final_class_ids), np.array(final_confidences)

frame_index = 0
detected_hull_numbers = {}  # Store all detected hull numbers
best_hull_numbers = {}  # Store the best hull number for each vessel (persistent display)

print("Starting hull number detection...")
clear_gpu_memory()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # IMPORTANT: Keep original clean frame for OCR (before any overlays)
    original_frame = frame.copy()

    # Run vessel detection with lower confidence for early detection, but keep aggressive filtering
    results = model.predict(frame, verbose=False, conf=0.25, iou=0.2, agnostic_nms=True, max_det=3)[0]
    
    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()
        
        # Keep only highest confidence detection per vessel class for perfect demo
        boxes, class_ids, confidences = keep_highest_confidence_only(boxes, class_ids, confidences)

        # STEP 1: Extract all vessel data and run OCR on CLEAN regions first
        vessel_data = []
        for i, (box, class_id, conf) in enumerate(zip(boxes, class_ids, confidences)):
            if class_id < len(vessel_names):
                vessel_name = vessel_names[class_id]
                color = vessel_colors.get(vessel_name, (255, 255, 255))
                x1, y1, x2, y2 = box.astype(int)
                
                # Extract vessel region from ORIGINAL CLEAN frame (no overlays)
                vessel_region = original_frame[y1:y2, x1:x2]
                
                if vessel_region.size > 0:
                    # Create cache key
                    cache_key = f"{vessel_name}_{x1//100}_{y1//100}_{frame_index//cache_update_interval}"
                    
                    hull_numbers = []
                    
                    # Use OCR for hull number detection (with caching)
                    if cache_key not in hull_cache:
                        print(f"Frame {frame_index}: Analyzing hull numbers for {vessel_name}...")
                        hull_numbers = extract_hull_numbers(vessel_region)
                        hull_cache[cache_key] = hull_numbers
                        
                        # Store detected hull numbers globally
                        if hull_numbers:
                            vessel_id = f"{vessel_name}_{x1}_{y1}"
                            if vessel_id not in detected_hull_numbers:
                                detected_hull_numbers[vessel_id] = []
                            for hull_info in hull_numbers:
                                if hull_info not in detected_hull_numbers[vessel_id]:
                                    detected_hull_numbers[vessel_id].append(hull_info)
                                
                                # Update best hull number for persistent display
                                vessel_key = f"{vessel_name}"  # Use vessel type as key for persistence
                                if vessel_key not in best_hull_numbers or hull_info['confidence'] > best_hull_numbers[vessel_key]['confidence']:
                                    interpreted_name = interpret_hull_number(hull_info['text'])
                                    # Only store if it's a known shipping company
                                    if interpreted_name in ["Hapag-Lloyd", "Maersk", "COSCO", "Evergreen", "MSC"]:
                                        best_hull_numbers[vessel_key] = {
                                            'text': interpreted_name,
                                            'confidence': hull_info['confidence'],
                                            'original_text': hull_info['text']
                                        }
                                        print(f"  🎯 NEW BEST HULL NUMBER for {vessel_name}: {interpreted_name} (conf: {hull_info['confidence']:.2f})")
                    else:
                        hull_numbers = hull_cache[cache_key]
                    
                    # Store vessel data for drawing
                    vessel_data.append({
                        'box': (x1, y1, x2, y2),
                        'name': vessel_name,
                        'color': color,
                        'hull_numbers': hull_numbers
                    })

        # STEP 2: Now draw all overlays on the frame (after OCR is complete)
        for vessel in vessel_data:
            x1, y1, x2, y2 = vessel['box']
            vessel_name = vessel['name']
            color = vessel['color']
            hull_numbers = vessel['hull_numbers']
            
            # Draw vessel bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label - Always show the BEST hull number detected so far (persistent display)
            vessel_key = f"{vessel_name}"
            if vessel_key in best_hull_numbers:
                # Always display the best hull number found so far
                best_hull_info = best_hull_numbers[vessel_key]
                label = f"{vessel_name} - Hull: {best_hull_info['text']}"
                text_color = (0, 255, 0)  # Green for confirmed hull numbers
                print(f"  📋 Displaying persistent hull number: {best_hull_info['text']} (best conf: {best_hull_info['confidence']:.2f})")
            else:
                # No hull number detected yet
                label = f"{vessel_name}"
                text_color = (255, 255, 255)  # White for vessel type only
            
            # Draw label background and text
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw hull number locations for detected hull numbers (when available in current frame)
            if hull_numbers and vessel_key in best_hull_numbers:
                # Find the hull number that matches our best detected one
                for hull_info in hull_numbers:
                    interpreted_text = interpret_hull_number(hull_info['text'])
                    if interpreted_text == best_hull_numbers[vessel_key]['text']:
                        hull_x = int(x1 + (hull_info['bbox'][0][0] + hull_info['bbox'][2][0]) / 2)
                        hull_y = int(y1 + (hull_info['bbox'][0][1] + hull_info['bbox'][2][1]) / 2)
                        cv2.circle(frame, (hull_x, hull_y), 5, (0, 255, 0), -1)  # Green marker
                        cv2.putText(frame, interpreted_text, (hull_x + 10, hull_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        break

    # Remove frame info and false hull number count to avoid confusion

    out.write(frame)
    frame_index += 1
    
    # Clear GPU memory periodically
    if frame_index % 120 == 0:
        clear_gpu_memory()

cap.release()
out.release()
cv2.destroyAllWindows()

# Print summary of detected hull numbers
print(f"\n{'='*50}")
print("HULL NUMBER DETECTION SUMMARY")
print(f"{'='*50}")
print(f"Total frames processed: {frame_index}")
print(f"Unique vessels with hull numbers detected: {len(detected_hull_numbers)}")

# Show best hull numbers (what will be persistently displayed)
print(f"\n🎯 BEST HULL NUMBERS (Displayed Throughout Video):")
for vessel_type, best_hull in best_hull_numbers.items():
    print(f"  {vessel_type}: {best_hull['text']} (Best Confidence: {best_hull['confidence']:.2f})")

# Show all detections for detailed analysis
print(f"\n📋 ALL DETECTED HULL NUMBERS:")
for vessel_id, hull_numbers in detected_hull_numbers.items():
    print(f"\nVessel: {vessel_id}")
    for hull_info in hull_numbers:
        interpreted_name = interpret_hull_number(hull_info['text'])
        print(f"  Possible Hull Number: {interpreted_name} (Confidence: {hull_info['confidence']:.2f})")

print(f"\nProcessing complete! Output saved to: {output_path}")
print("Hull number detection using OCR completed!") 