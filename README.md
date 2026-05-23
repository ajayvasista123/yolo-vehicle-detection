# YOLOv8 Vehicle Detection Pipeline

Object detection pipeline using YOLOv8 that detects and localises vehicles in real-time with bounding boxes and confidence scores.

## Demo
Real-time detection on live street footage — car detected at 0.91 confidence at 14-16 fps on a standard laptop CPU.

## What it does
- Real-time vehicle detection from live camera feed at 14-16 fps
- Batch detection on car damage images (dents, scratches, collision damage)
- Outputs bounding boxes with confidence scores for each detected object
- Inference time: 62-163ms per image

## Scripts
- detect.py — single image detection
- detect_all.py — batch detection on multiple images
- realtime_detect.py — live webcam detection

## Stack
Python, YOLOv8, OpenCV, Linux, Windows

## How to run
pip install ultralytics opencv-python
python realtime_detect.py
