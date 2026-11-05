# Vehicle Traffic Analytics Pipeline

A real-time computer vision pipeline for top-down vehicle tracking and counting. This repository demonstrates the transition from a high-parameter cloud validation model to an edge-optimized deployment architecture.

## System Architecture

* **Validation Tier (Ground Truth):** YOLOv8x (68.2M parameters). Trained via cloud compute to establish a strict ground-truth baseline of 29 total vehicles (13 Inbound, 16 Outbound) on the validation video.
* **Edge Tier (Production):** YOLOv8n (3.2M parameters). Quantized and exported to ONNX (FP16, ~6MB). Matches the 29-vehicle ground-truth baseline while running at real-time FPS on local CPU hardware.

## Pipeline Features

* **Heuristic Spatial Filtering:** Utilizes region-of-interest (ROI) masking (`xyxy[:, 1] > 150`) to decouple static false-positives (e.g., road infrastructure) from dynamic tracking logic.
* **Tracking Algorithm:** ByteTrack. Selected over BoT-SORT to maintain ID retention on low-confidence edge detections and minimize computational overhead.
* **Interface:** Streamlit-based dashboard supporting local video file processing and real-time metric visualization.

## Repository Structure

```text
├── app.py                   # Streamlit UI & inference loop
├── README.md                
├── requirements.txt         
├── assets/                  # Training metrics & validation charts
├── data/                    # Default sample video
├── models/                  
│   └── best.onnx            # 6MB Edge-optimized ONNX model
└── uploads/                 # Temporary directory for video processing