import cv2
from sort import *
import math
import numpy as np
from ultralytics import YOLO
import cvzone

click_points = []
line = []  # Tự động cập nhật sau khi chọn 2 điểm
counter = []


def mouseClick(event, x, y, flags, params):
    global click_points, line
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(click_points) < 2:
            click_points.append((x, y))
            print(f"Điểm {len(click_points)}: ({x}, {y})")
        if len(click_points) == 2:
            line = [click_points[0][0], click_points[0][1], click_points[1][0], click_points[1][1]]
            print(f"\n👉 Line đã sẵn sàng: {line}")

# Load video và model
# cap = cv2.VideoCapture('car.mp4')
cap = cv2.VideoCapture('Video xe máy - xe hơi chạy trên đường.mp4')
model = YOLO('best.pt')

# Gán callback chuột
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", mouseClick)

# Đọc danh sách class
classnames = []
with open('class.txt', 'r') as f:
    classnames = f.read().splitlines()

# Tracker
tracker = Sort(max_age=20)

while True:
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (640, 360))

    if not ret:
        # cap = cv2.VideoCapture('car.mp4')
        cap = cv2.VideoCapture('Video xe máy - xe hơi chạy trên đường.mp4')
        continue

    # Vẽ các điểm click chuột (chọn line)
    for point in click_points:
        cv2.circle(frame, point, 5, (255, 0, 0), -1)
    if len(click_points) == 2:
        cv2.line(frame, click_points[0], click_points[1], (0, 255, 255), 3)

    detections = np.empty((0, 5))
    result = model(frame, stream=True)

    for info in result:
        boxes = info.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            classindex = box.cls[0]
            conf = math.ceil(conf * 100)
            classindex = int(classindex)
            objectdetect = classnames[classindex]

            if objectdetect in ['0', '1','2','3', 'car', 'bus', 'truck','motor'] and conf > 40:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                new_detections = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, new_detections))

    track_result = tracker.update(detections)

    # Nếu đã có line thì mới thực hiện đếm
    if len(line) == 4:
        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), 7)

        for results in track_result:
            x1, y1, x2, y2, id = results
            x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cvzone.putTextRect(frame, f'{id}', [x1 + 8, y1 - 12], thickness=2, scale=1.5)

            # Kiểm tra cắt qua line
            if line[0] < cx < line[2] and line[1] - 20 < cy < line[1] + 20:
                cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 15)
                if id not in counter:
                    counter.append(id)

    # Hiển thị tổng số xe
    cvzone.putTextRect(frame, f'Total Vehicles = {len(counter)}', [290, 34], thickness=4, scale=2.3, border=2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27:  # Nhấn ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()
