import cv2
import os

video_path = "videos/"
output_dir = "frames/"
fps_extract = 0.7  # frames per second

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)

frame_interval = int(video_fps / fps_extract)
count = 0
saved = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Done")
        break

    if count % frame_interval == 0:
        cv2.imwrite(f"{output_dir}/frame_{saved:05d}.jpg", frame)
        saved += 1

    count += 1
    print(f"Extracted {saved} frames")

cap.release()
print(f"Saved {saved} frames")