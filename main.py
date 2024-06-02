import cv2
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('/Users/lsm99/내 드라이브/기계학습/keyboard-mouse-monitor/weights/best.pt')

# 클래스 이름 설정 (모델 학습 시 사용한 클래스 이름과 동일해야 함)
class_names = ["keyboard", "monitor", "mouse"]  # 여기에 클래스 이름을 추가

# 입력 비디오 파일 경로 설정
input_video_path = '/Users/lsm99/내 드라이브/기계학습/keyboard-mouse-monitor/input_video.mp4'
output_video_path = '/Users/lsm99/내 드라이브/기계학습/keyboard-mouse-monitor/output_video.mp4'

# 비디오 로드
cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 비디오 라이터 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 객체 탐지 수행
    results = model(frame)

    # 프레임에 바운딩 박스 및 레이블 그리기
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].tolist()
                class_id = int(box.cls[0].tolist())  # 클래스 ID 가져오기
                label = f'{class_names[class_id]} {confidence:.2f}'  # 클래스 이름과 신뢰도 포함

                # 바운딩 박스와 라벨 그리기
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # 결과 프레임을 출력 동영상에 작성
    out.write(frame)

# 작업 완료 후 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as {output_video_path}")
