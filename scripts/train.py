from ultralytics import YOLO
import os

# ── Configuration ──────────────────────────────────────────
BASE         = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE, "data/surgical-instruments/data.yaml")
MODEL        = "yolov8n.pt"   # starting from pretrained weights
EPOCHS       = 50             # how many times to study the dataset
IMAGE_SIZE   = 640            # standard input size for YOLOv8
PROJECT      = os.path.join(BASE, "runs/detect/models")  # where to save results
RUN_NAME     = "surgical-v1"  # name of this training run
# ───────────────────────────────────────────────────────────

def train():
    print("Loading base model...")
    model = YOLO(MODEL)

    print(f"Starting training for {EPOCHS} epochs...")
    results = model.train(
        data     = DATASET_PATH,
        epochs   = EPOCHS,
        imgsz    = IMAGE_SIZE,
        project  = PROJECT,
        name     = RUN_NAME,
        patience = 10,        # stops early if no improvement
        batch    = 8,         # images processed at once
        verbose  = True
    )

    print("Training complete!")
    print(f"Best model saved to: {PROJECT}/{RUN_NAME}/weights/best.pt")

if __name__ == "__main__":
    train()