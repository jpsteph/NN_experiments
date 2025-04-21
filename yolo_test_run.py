from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO(r"C:\Users\LENOVO\Documents\_Python Projects\yolo_test\new_data_set\dataP2.yaml").load("yolo11n.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data=r"C:\Users\LENOVO\Documents\_Python Projects\yolo_test\new_data_set\dataP2.yaml",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    device="cpu",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)
