# YOLOv8 Vehicle Detection Pipeline

Object detection pipeline using YOLOv8 that detects and localises vehicles in images with bounding boxes and confidence scores. Tested on real car damage images.

## What it does
- Detects vehicles in images using YOLOv8 neural network
- Outputs bounding boxes with confidence scores for each detected object
- Processes images in 62-163ms per image
- Tested on real damaged car images (dents, scratches, collision damage)

## Stack
Python, YOLOv8, OpenCV, Linux

## How to run
pip install ultralytics opencv-python-headless
python3 detect_all.py
