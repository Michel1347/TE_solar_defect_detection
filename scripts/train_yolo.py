"""
☀️ Solar Panel Detection using YOLOv8

This script provides reusable functions to:

1. Train a YOLOv8 object detection model on a custom solar panel dataset

The model detects only one class:
    - solar_panel
"""

from ultralytics import YOLO
import os


# ==========================================================
# 🚀 TRAINING FUNCTION
# ==========================================================
def train_solar_panel_model(
    dataset_yaml: str,
    base_model: str = "yolov8s.pt",
    epochs: int = 100,
    imgsz: int = 512,
    batch: int = 4,
    output_dir: str = "runs/detect/train"
):
    """
    Train YOLOv8 solar panel detection model.

    Parameters
    ----------
    dataset_yaml : str
        Path to your dataset config file (data.yaml).
        Example: "solar_panel_dataset/data.yaml"

    base_model : str
        Starting pretrained YOLO model.
        Options:
            - yolov8n.pt (fastest, lowest accuracy)
            - yolov8s.pt (balanced)
            - yolov8m.pt (higher accuracy, needs more GPU)

    epochs : int
        Number of full training cycles.
        More epochs → better learning but longer training time.

    imgsz : int
        Image resolution used during training.
        Higher value → better detection accuracy but slower training.
        Common values: 416, 512, 640

    batch : int
        Number of images processed per training step.
        Higher batch → faster training but requires more VRAM.

    output_dir : str
        Folder where YOLO saves training results.

    Returns
    -------
    str
        Path to the best trained weights file (best.pt).
    """

    print("\n🚀 Starting YOLOv8 Training...\n")

    # Load pretrained YOLO model
    model = YOLO(base_model)

    # Train model on custom dataset
    model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=output_dir,
        name="solar_panel_training"
    )

    # Best model weights are automatically saved here
    best_model_path = os.path.join(
        output_dir,
        "solar_panel_training",
        "weights",
        "best.pt"
    )

    print("\n✅ Training Completed Successfully!")
    print("📌 Best Model Saved At:", best_model_path)

    return best_model_path

# ==========================================================
# 🧪 EXAMPLE SCRIPT EXECUTION
# ==========================================================
if __name__ == "__main__":

    # -----------------------------
    # Train Model
    # -----------------------------
    best_model = train_solar_panel_model(
        dataset_yaml="solar_panel_dataset/data.yaml",
        epochs=100,
        imgsz=512,
        batch=4
    )
