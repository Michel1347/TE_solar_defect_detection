"""
☀️ Solar Panel Detection using YOLOv8

This script provides reusable functions to:

1. Run prediction (detection) on images, videos, or folders
2. Return structured detection results + output file paths

The model detects only one class:
    - solar_panel
"""

from ultralytics import YOLO
import os

# ==========================================================
# 🔍 DETECTION / PREDICTION FUNCTION
# ==========================================================
def detect_solar_panels(
    model_path: str,
    source_path: str,
    conf: float = 0.2,
    iou: float = 0.5,
    save_results: bool = True
):
    """
    Detect solar panels in an image/video/folder.

    Parameters
    ----------
    model_path : str
        Path to trained YOLO model weights.
        Example: "runs/detect/train/weights/best.pt"

    source_path : str
        Input file or folder to run detection on.
        Examples:
            - "test.jpg"
            - "drone_video.mp4"
            - "test_images/" (detect all files in folder)

    conf : float
        Confidence threshold (0.0 → 1.0)

        - Low conf (0.1–0.3): detects more objects but may include false positives
        - High conf (0.6–0.9): detects fewer objects but more accurate

    iou : float
        Intersection-over-Union threshold for filtering overlapping boxes.

        - Lower IOU (0.3–0.5): keeps more boxes
        - Higher IOU (0.7–0.9): removes duplicates aggressively

    save_results : bool
        If True → YOLO saves annotated output images/videos automatically.

    Returns
    -------
    dict
        Dictionary containing:
            - predictions (bounding boxes + confidence)
            - saved_files (paths to annotated output images/videos)
    """

    print("\n🔍 Loading YOLO Model...\n")
    model = YOLO(model_path)

    print(f"📂 Running Detection on: {source_path}")

    # Run prediction
    results = model.predict(
        source=source_path,
        conf=conf,
        iou=iou,
        save=save_results
    )

    predictions = []
    saved_files = []

    # Process detection results
    for r in results:

        # Save output file path if YOLO saved annotated result
        if save_results:
            saved_files.append(r.save_dir)

        # Extract bounding boxes
        for box in r.boxes:
            predictions.append({
                "class_id": int(box.cls),                # class index
                "confidence": float(box.conf),           # detection confidence
                "bbox_xyxy": box.xyxy.tolist()           # bounding box coords
            })

    print("\n✅ Detection Complete!")

    return {
        "predictions": predictions,
        "saved_files": list(set(saved_files))  # remove duplicates
    }


# ==========================================================
# 🧪 EXAMPLE SCRIPT EXECUTION
# ==========================================================
if __name__ == "__main__":

    # -----------------------------
    # Run Detection
    # -----------------------------
    output = detect_solar_panels(
        model_path=best_model,
        source_path="frames/",
        conf=0.25,
        iou=0.5
    )

    print("\n📌 Predictions Found:")
    print(output["predictions"])

    print("\n🖼️ Annotated Output Saved In:")
    print(output["saved_files"])