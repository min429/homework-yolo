import cv2
from ultralytics import YOLO

model = YOLO('/Users/lsm99/내 드라이브/기계학습/keyboard-mouse-monitor/weights/best.pt')

class_names = ["keyboard", "monitor", "mouse"]

input_video_path = '/Users/lsm99/내 드라이브/기계학습/keyboard-mouse-monitor/input_video.mp4'
output_video_path = '/Users/lsm99/내 드라이브/기계학습/keyboard-mouse-monitor/output_video.mp4'

cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].tolist()
                class_id = int(box.cls[0].tolist()) 
                label = f'{class_names[class_id]} {confidence:.2f}' 

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
