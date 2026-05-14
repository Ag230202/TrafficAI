"""
train_yolo_emergency.py
-----------------------
Fine-tunes the YOLOv8 nano model (yolov8n.pt) on a custom dataset
to natively detect 'ambulance' and 'fire_truck' classes.

Instructions:
1. Ensure you have a YOLO-formatted dataset (images and labels).
2. Create a 'dataset.yaml' file pointing to your train/val sets
   and defining the class names:
   
   # dataset.yaml example:
   path: ./datasets/emergency_vehicles
   train: images/train
   val: images/val
   names:
     0: car
     1: motorcycle
     2: bus
     3: truck
     4: ambulance
     5: fire_truck

3. Run this script:
   python train_yolo_emergency.py

4. Once trained, point 'model_path' in detector.py to the new weights
   (e.g., runs/detect/train/weights/best.pt).
"""

from ultralytics import YOLO
import os

def main():
    # 1. Load the pre-trained nano model
    model_path = "yolov8n.pt"
    if not os.path.exists(model_path):
        print(f"Downloading {model_path}...")
    
    model = YOLO(model_path)

    # 2. Path to your dataset configuration
    data_yaml = "dataset.yaml"
    
    if not os.path.exists(data_yaml):
        print(f"[ERROR] {data_yaml} not found.")
        print("Please create a dataset.yaml file with your dataset paths and classes.")
        print("See the script docstring for an example.")
        return

    # 3. Start fine-tuning
    print("Starting YOLOv8 fine-tuning for emergency vehicles...")
    
    # You can tweak these hyperparameters based on your GPU capability
    results = model.train(
        data=data_yaml,
        epochs=50,             # Number of training epochs
        imgsz=640,             # Image size for training
        batch=16,              # Batch size (lower if running out of memory)
        device="cpu",          # Set to '0' if you have an NVIDIA GPU, otherwise 'cpu'
        patience=10,           # Early stopping if no improvement
        name="yolov8_emergency"# Folder name inside runs/detect/
    )

    print("\n[SUCCESS] Training complete!")
    print("The best weights are saved in: runs/detect/yolov8_emergency/weights/best.pt")
    print("Update detector.py to use this new weight file.")

if __name__ == "__main__":
    main()
