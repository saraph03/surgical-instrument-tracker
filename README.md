# 🔬 Surgical Instrument Tracker

A computer vision system that detects and tracks surgical instruments in real-time video to assist doctors and robotic surgical systems.

Built as a portfolio project in preparation for the MS in Robotics program at Northeastern University (Khoury College of Computer Sciences).

---

## 🎯 Project Goal

To develop an AI-powered pipeline that can:
- Detect surgical instruments in laparoscopic video footage
- Track instrument movement across frames
- Alert when instruments enter defined danger zones
- Feed positional data into a simulated robotic assistant

---

## 🛠️ Tech Stack

- **YOLOv8** — real-time object detection
- **OpenCV** — video processing
- **PyTorch** — deep learning engine
- **PyBullet** — robotic arm simulation (Month 3)
- **ByteTrack** — multi-object tracking (Month 2)

---

## 📅 Development Roadmap

- [x] Month 1 — Instrument detection on surgical video
- [ ] Month 2 — Real-time tracking & danger zone alerts
- [ ] Month 3 — Robotic arm simulation integration
- [ ] Month 4 — Surgical phase classification
- [ ] Month 5 — Full demo & documentation

---

## 📂 Project Structuresurgical-instrument-tracker/
├── data/          # Surgical video datasets
├── models/        # Trained model weights
├── scripts/       # Core Python scripts
├── results/       # Detection outputs
├── notebooks/     # Experiments & analysis
└── README.md
---

## 🚀 Getting Started
```bash
git clone https://github.com/YOUR_USERNAME/surgical-instrument-tracker.git
cd surgical-instrument-tracker
python3 -m venv venv
source venv/bin/activate
pip install ultralytics opencv-python torch
```

---

## 📊 Dataset

Using the **Cholec80** dataset — 80 laparoscopic cholecystectomy videos with surgical instrument annotations.

---

## 👩‍💻 Author

Sara Phondge — aspiring robotics engineer focused on medical device innovation.
