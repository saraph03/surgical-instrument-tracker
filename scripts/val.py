from ultralytics import YOLO
import os

# ── Configuration ──────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "runs/detect/models/surgical-v12/weights/best.pt")
DATA_PATH  = os.path.join(BASE, "data/surgical-instruments/data.yaml")
IMAGE_SIZE = 640
# ───────────────────────────────────────────────────────────

def validate():
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print("Running validation...")
    metrics = model.val(
        data  = DATA_PATH,
        imgsz = IMAGE_SIZE,
    )

    # ── Overall metrics ────────────────────────────────────
    print("\n── Overall Results ───────────────────────────────")
    print(f"  mAP50:       {metrics.box.map50:.3f}")
    print(f"  mAP50-95:    {metrics.box.map:.3f}")
    print(f"  Precision:   {metrics.box.mp:.3f}")
    print(f"  Recall:      {metrics.box.mr:.3f}")

    # ── Per-class breakdown ────────────────────────────────
    print("\n── Per-Class mAP50 ───────────────────────────────")
    names = model.names
    for i, ap in enumerate(metrics.box.ap50):
        label = names[i] if names else f"class {i}"
        bar   = "█" * int(ap * 20)
        print(f"  {label:<15} {ap:.3f}  {bar}")

    print("\nValidation complete!")
    print(f"Full results saved to: runs/detect/val")

if __name__ == "__main__":
    validate()