# ☀️ Solar Panel Detection using YOLOv8

A custom-trained **YOLOv8 object detection system** for detecting **solar panels** in images and videos, including drone footage.

The project uses a custom dataset to fine-tune a YOLOv8s model for a single object class:

- `solar_panel`

The trained model achieves **72.1% mAP@0.5** and is designed to provide a practical balance between detection accuracy and inference speed.

---

## 🚀 Project Overview

The project provides a simple pipeline for detecting solar panels from images and video:

Video
  │
  ▼
extracted_frames.py
  │
  │ Extract frames at configurable FPS
  ▼
frames/
  │
  ▼
run_detection.py
  │
  │ YOLOv8 inference
  ▼
Detected solar panels
  │
  ▼
Annotated images/videos

 ---

 ## 🚀 Model Performance

 | Metric | Value |
 |------|------|
 | **Model Architecture** | YOLOv8s |
 | **mAP@0.5** | **72.1%** |
 | Input Resolution | 512 × 512 |
 | Task | Object Detection |
 | Class | solar_panel |

 > The model balances **accuracy and speed**, making it suitable for systems with limited GPU memory (4–6 GB VRAM).

 ---

🧠 Model Training

The model was trained using Ultralytics YOLOv8 with a custom solar-panel dataset.

Training command
yolo detect train model=yolov8s.pt data=solar_panel_dataset/data.yaml epochs=100 imgsz=512 batch=4
Training parameters
Parameter	Value
Base model	yolov8s.pt
Epochs	100
Image size	512 × 512
Batch size	4
Task	Detection

After training, the best model weights are typically stored at:

runs/detect/train/weights/best.pt
🔍 Running Detection

There are two ways to run detection.

Option 1 — YOLO CLI

To detect solar panels using the trained model:

yolo detect predict \
    model="runs/detect/train/weights/best.pt" \
    source="frames/" \
    conf=0.2 \
    iou=0.5
Parameters

model

Path to the trained YOLO model:

runs/detect/train/weights/best.pt

source

Can be an image, video, or directory:

image.jpg
video.mp4
frames/

conf

Confidence threshold:

0.2

Lower values detect more objects but may increase false positives.

Higher values produce fewer detections but generally require stronger model confidence.

iou

Intersection-over-Union threshold used during non-maximum suppression:

0.5
🐍 Reusable Python Detection Script

run_detection.py provides a reusable Python function:

detect_solar_panels(
    model_path,
    source_path,
    conf=0.2,
    iou=0.5,
    save_results=True
)

The function can process:

Individual images
Videos
Directories containing multiple images/videos
Example
output = detect_solar_panels(
    model_path="runs/detect/train/weights/best.pt",
    source_path="frames/",
    conf=0.25,
    iou=0.5
)


print(output["predictions"])
print(output["saved_files"])
Returned results

The function returns a dictionary containing:

{
    "predictions": [...],
    "saved_files": [...]
}

Each prediction contains:

{
    "class_id": 0,
    "confidence": 0.87,
    "bbox_xyxy": [...]
}

This makes the detection function useful for integrating the model into larger Python applications or APIs.

🎥 Video Frame Extraction

extracted_frames.py extracts individual frames from video files using OpenCV.

This is useful when drone footage needs to be converted into individual images before running detection.

Configuration
video_path = "videos/"
output_dir = "frames/"
fps_extract = 0.7

The fps_extract variable controls how many frames are extracted per second.

For example:

fps_extract = 0.7

means approximately 0.7 frames per second are extracted.

The extracted frames are saved using sequential filenames:

frames/
├── frame_00000.jpg
├── frame_00001.jpg
├── frame_00002.jpg
└── ...
🔄 Complete Video Detection Workflow

For drone footage, the complete workflow can be:

1. Place the video in the input directory
videos/
└── drone_video.mp4
2. Extract frames

Run:

python scripts/extracted_frames.py

This generates:

frames/
├── frame_00000.jpg
├── frame_00001.jpg
├── frame_00002.jpg
└── ...
3. Run detection
python scripts/run_detection.py

Or use the YOLO CLI:

yolo detect predict \
    model="runs/detect/train/weights/best.pt" \
    source="frames/" \
    conf=0.2 \
    iou=0.5
4. Review the annotated results

Ultralytics will save the annotated detection results in its generated output directory.

⚙️ Installation

Clone the repository:

git clone https://github.com/Michel1347/TE_solar_defect_detection.git
cd YOUR_REPOSITORY

Create a virtual environment:

python -m venv .venv

Activate it on Linux:

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

The main libraries used by the project are:

Ultralytics YOLOv8
OpenCV
Python
🖥️ Hardware Considerations

YOLOv8s was selected to provide a reasonable trade-off between:

Detection accuracy
Inference speed
GPU memory requirements

The model can be used on systems with relatively limited GPU resources, although inference performance will depend on the hardware and input resolution.

🛠️ Future Improvements

 Detect solar panels directly from video streams
 Track solar panels across consecutive video frames
 Calculate the number of detected panels
 Estimate panel area from bounding boxes
 Improve detection performance with a larger dataset
 Evaluate precision, recall, and mAP at different confidence thresholds
 Build a web interface for uploading images/videos
 Expose the detector through a FastAPI endpoint
 Deploy the model as an automated solar-inspection service


📌 Use Cases

Potential applications include:

☀️ Solar farm inspection
🚁 Drone-based solar panel surveys
🏠 Rooftop solar detection
🛰️ Aerial imagery analysis
🔍 Automated solar asset identification
📊 Solar infrastructure mapping
