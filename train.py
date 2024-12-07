from ultralytics import YOLO

# Step 1: Load YOLO model
# Replace 'yolov8n.pt' with 'yolov8s.pt', 'yolov8m.pt', etc., if you need a larger model
model = YOLO('yolov8n.pt')  # Start with a pre-trained YOLOv8 nano model

# Step 2: Train the model
# Ensure the fire_detection.yaml file exists with proper paths to your dataset
model.train(
    data='C:/Users/User/Downloads/Compressed/fire_detection.yaml',  # Dataset config path
    epochs=50,  # Number of training epochs
    imgsz=640,  # Image size
    batch=16,   # Batch size
    name='fire_detection',  # Save experiment results with this name
    device=0    # Use GPU (set -1 for CPU)
)

# Step 3: Validate the model
metrics = model.val()  # Evaluate model performance on the validation dataset
print("Validation metrics:", metrics)

# Step 4: Predict on test images
results = model.predict(
    source='C:/Users/User/Downloads/Compressed/test/images',  # Path to test images
    save=True  # Save predictions in a folder (default is runs/predict/exp)
)

# Step 5: Export the trained model (optional, for deployment)
model.export(format='onnx')  # Export to ONNX format for deployment
