import cv2
from ultralytics import YOLO
import os

# ── Configuration ──────────────────────────────────────────
VIDEO_PATH = "data/sample_video.mp4"   
OUTPUT_DIR = "results/"
CONFIDENCE = 0.25                       # only show detections above 25% confidence
# ───────────────────────────────────────────────────────────

def run_detection(video_path, output_dir, confidence):
    """
    Runs YOLOv8 object detection on a video file.
    Draws bounding boxes on each frame and saves the output video.
    """

    # Load the YOLOv8 model (downloads automatically if not present)
    print("Loading model...")
    model = YOLO("yolov8n.pt")

    # Open the video file
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Error: Could not open video. Check your file path.")
        return

    # Get video properties so our output matches the input
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))

    # Set up the output video writer
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "detection_output.mp4")
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    print("Running detection... press Ctrl+C to stop early.")
    frame_count = 0

    while True:
        ret, frame = cap.read()

        # Stop if video ends
        if not ret:
            break

        # Run detection on this frame
        results = model(frame, conf=confidence, verbose=False)

        # Draw bounding boxes onto the frame
        annotated_frame = results[0].plot()

        # Save the annotated frame to our output video
        writer.write(annotated_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count} frames...")

    # Clean up
    cap.release()
    writer.release()
    print(f"✅ Done! Output saved to: {output_path}")
    print(f"   Total frames processed: {frame_count}")


if __name__ == "__main__":
    run_detection(VIDEO_PATH, OUTPUT_DIR, CONFIDENCE)