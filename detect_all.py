from ultralytics import YOLO
import time

model = YOLO('yolov8n.pt')
images = ['car1.jpg', 'car2.jpg', 'car3.jpg']

for img in images:
    start = time.time()
    results = model(img)
    end = time.time()
    result = results[0]
    print(f"\n--- {img} ---")
    print(f"Inference time: {(end-start)*1000:.1f}ms")
    print(f"Objects found: {len(result.boxes)}")
    for box in result.boxes:
        name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        print(f"  {name}: {conf:.2f}")
    result.save(filename=f"output_{img}")
    print(f"Saved output_{img}")

print("\nALL DONE")
