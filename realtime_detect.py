import cv2
from ultralytics import YOLO
import time

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting real-time detection... Press Q to quit")

fps_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    results = model(frame, verbose=False)
    end = time.time()

    fps = 1 / (end - start)
    fps_list.append(fps)

    annotated = results[0].plot()
    cv2.putText(annotated, f'FPS: {fps:.1f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('YOLOv8 Real-Time Detection - Ajay Vasista', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

avg_fps = sum(fps_list) / len(fps_list)
print(f"Average FPS: {avg_fps:.1f}")
print(f"Max FPS: {max(fps_list):.1f}")