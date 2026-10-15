# Multi-Object Tracking in Sports Videos

This project detects and tracks players in sports videos using deep learning. A YOLOv8 object detector is fine-tuned on the [SportsMOT](https://github.com/MCG-NJU/SportsMOT) dataset, and two popular tracking algorithms, **ByteTrack** and **DeepSORT**, are compared on top of it. The goal is to find every player in each frame and keep a consistent ID for each one throughout the video.

The project was developed and run on [Kaggle](https://www.kaggle.com/) notebooks.

## What I Did

1. Prepared the SportsMOT annotations for training a detector.
2. Fine-tuned a pretrained YOLOv8 model to detect players.
3. Added ByteTrack and DeepSORT to track players across frames and generate annotated videos.
4. Evaluated both trackers against the ground truth using standard tracking metrics (IDF1, MOTA, Precision).

## Results

Average scores over 5 validation videos:

| Tracker | IDF1 | MOTA | Precision |
|---------|------|------|-----------|
| **ByteTrack** | **0.621** | **0.805** | **0.916** |
| DeepSORT | 0.509 | 0.351 | 0.643 |

ByteTrack clearly outperformed DeepSORT in this setup.

## Repository Contents

- `yolo.ipynb`: dataset preparation and detector fine-tuning
- `motbytetrack.ipynb`: tracking with ByteTrack and DeepSORT, plus evaluation
