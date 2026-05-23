from ultralytics import YOLO
import time

model = YOLO('yolov8n.pt')

start = time.time()
results = model("test.jpg")
end = time.time()

result = results[0]
print(f"Inference time: {(end-start)*1000:.1f}ms")
print(f"Objects found: {len(result.boxes)}")
for box in result.boxes:
    name = model.names[int(box.cls[0])]
    conf = float(box.conf[0])
    print(f"  {name}: {conf:.2f}")

result.save(filename="car_output.jpg")
print("Saved car_output.jpg")
