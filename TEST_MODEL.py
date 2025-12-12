import os
from ultralytics import YOLO
from IPython.display import Image, display

# Configuration
MODEL_PATH = "blood_cell_project/yolov8_blood_cells_v1/weights/best.pt"  
TEST_IMAGES_DIR = "datasets/test/images" 
OUTPUT_DIR = "blood_cell_project/test_results"  
CONF_THRESHOLD = 0.25 

# Make sure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load trained YOLOv8 model
model = YOLO(MODEL_PATH)
print(f"Loaded model: {MODEL_PATH}")

# Run inference on each image
for img_file in os.listdir(TEST_IMAGES_DIR):
    if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        print(f"\nProcessing {img_file}...")
        
        # Predict
        results = model.predict(source=img_path, conf=CONF_THRESHOLD, save=True, save_dir=OUTPUT_DIR)
        
        # Display detection info
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            print(f"Class: {cls}, Confidence: {conf:.2f}, BBox: {xyxy}")
        
        # Display annotated image
        annotated_path = os.path.join(OUTPUT_DIR, img_file)
        if os.path.exists(annotated_path):
            display(Image(filename=annotated_path))

