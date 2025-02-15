import cv2
import torch
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  

camera = cv2.VideoCapture(0)  

while True:
    ret, frame = camera.read()
    if not ret:
        break

    results = model(frame)

    annotated_frame = results[0].plot()  

    cv2.imshow("Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()




