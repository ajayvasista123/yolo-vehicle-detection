import cv2
from ultralytics import YOLO
import time

# Load model
model = YOLO('yolov8n.pt')

# Run detection on an image
print("Downloading test image...")
import urllib.request
urllib.request.urlretrieve(
    "https://ultralytics.com/images/bus.jpg",
    "test.jpg"
)

# Run inference
print("Running detection...")
start = time.time()
results = model("test.jpg")
end = time.time()

# Process results
result = results[0]
boxes = result.boxes

print(f"\n--- Detection Results ---")
print(f"Objects detected: {len(boxes)}")
print(f"Inference time: {(end-start)*1000:.1f}ms")
print(f"\nDetected objects:")

for box in boxes:
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    confidence = float(box.conf[0])
    coords = box.xyxy[0].tolist()
    print(f"  {class_name}: {confidence:.2f} confidence at {[round(c,1) for c in coords]}")

# Save output image with boxes drawn
result.save(filename="output.jpg")
print(f"\nOutput image saved as output.jpg")
print("SUCCESS - YOLOv8 detection pipeline working!")

