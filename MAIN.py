# traffic_optimization_scoot.py
import cv2
from sort import *
import math
import numpy as np
import time
from ultralytics import YOLO
import cvzone

# ------------------------ Khởi tạo ------------------------
click_points = []
lines = []  # 3 line tương ứng 3 hướng
zones_draw = []  # lưu vùng queue bằng click chuột phải (2 điểm tạo zone)
queue_zones = []  # vùng hàng đợi thực tế
max_directions = 3

counters = [[] for _ in range(max_directions)]
vehicle_times = [{} for _ in range(max_directions)]
vehicle_speeds = [{} for _ in range(max_directions)]
vehicle_positions = [{} for _ in range(max_directions)]
headways = [[] for _ in range(max_directions)]
queue_counters = [[] for _ in range(max_directions)]
last_red_time = [0 for _ in range(max_directions)]
estimated_green = [10 for _ in range(max_directions)]

min_green_time = [10, 10, 10]
max_green_time = [45, 45, 45]
traffic_light_status = ["RED", "RED", "RED"]
green_start_time = [0, 0, 0]

fps = 30
queue_threshold_speed = 3
headway_threshold = 3

# ------------------------ Hàm phụ trợ ------------------------
def mouseClick(event, x, y, flags, params):
    global click_points, lines, zones_draw, queue_zones
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((x, y))
        print(f"LINE point: ({x}, {y})")
        if len(click_points) % 2 == 0 and len(click_points) <= max_directions * 2:
            lines = [
                [click_points[i][0], click_points[i][1], click_points[i + 1][0], click_points[i + 1][1]]
                for i in range(0, len(click_points), 2)
            ]
            print("Lines:", lines)
    elif event == cv2.EVENT_RBUTTONDOWN:
        zones_draw.append((x, y))
        print(f"ZONE point: ({x}, {y})")
        if len(zones_draw) % 2 == 0:
            queue_zones = [
                [
                    (zones_draw[i][0], zones_draw[i][1], zones_draw[i + 1][0], zones_draw[i + 1][1])
                ]
                for i in range(0, len(zones_draw), 2)
            ]
            print("Queue Zones:", queue_zones)

def check_vehicle_in_queue_zone(x, y, zone):
    x1, y1, x2, y2 = zone
    return x1 <= x <= x2 and y1 <= y <= y2

def calculate_optimal_green_time(i):
    q_len = len(queue_counters[i])
    avg_hw = sum(headways[i]) / len(headways[i]) if headways[i] else 2
    base_green = max(min_green_time[i], q_len * avg_hw)
    vol_factor = max(len(counters[i]) / 10, 0.5)
    avg_spd = sum(vehicle_speeds[i].values()) / len(vehicle_speeds[i]) if vehicle_speeds[i] else 0
    spd_factor = 1.0 if avg_spd < 5 else (0.8 if avg_spd < 15 else 0.6)
    green_time = base_green * vol_factor * spd_factor
    return int(max(min_green_time[i], min(max_green_time[i], green_time)))

def calculate_direction_priority():
    priorities = []
    for i in range(max_directions):
        queue_factor = len(queue_counters[i]) * 2
        wait_factor = min((time.time() - last_red_time[i]) / 30, 3)
        priorities.append((i, queue_factor * wait_factor))
    return sorted(priorities, key=lambda x: x[1], reverse=True)

def simulate_traffic_lights(now):
    global traffic_light_status, green_start_time, estimated_green, last_red_time
    current = -1
    for i in range(max_directions):
        if traffic_light_status[i] == "GREEN":
            current = i
            if now - green_start_time[i] >= estimated_green[i]:
                traffic_light_status[i] = "RED"
                last_red_time[i] = now
                counters[i] = []
                current = -1
    if current == -1:
        next_i = calculate_direction_priority()[0][0]
        traffic_light_status[next_i] = "GREEN"
        green_start_time[next_i] = now
        estimated_green[next_i] = calculate_optimal_green_time(next_i)
        queue_counters[next_i] = []

def display_traffic_info(frame):
    font_scale, thickness, lh = 0.6, 1, 25
    start_x, start_y = 10, 30
    panel_w, panel_h = 250, max_directions * lh * 3 + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (start_x - 5, start_y - 25), (start_x + panel_w, start_y + panel_h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    for i in range(max_directions):
        y = start_y + i * lh * 3
        cv2.putText(frame, f"Huong {chr(65+i)}: {len(counters[i])} xe | Hang doi: {len(queue_counters[i])}", (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        spd = sum(vehicle_speeds[i].values()) / len(vehicle_speeds[i]) if vehicle_speeds[i] else 0
        cv2.putText(frame, f"Speed TB: {spd:.1f}px/s", (start_x, y+lh), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)
        if traffic_light_status[i] == "GREEN":
            remaining = estimated_green[i] - (time.time() - green_start_time[i])
            cv2.putText(frame, f"DEN XANH: {int(remaining)}s", (start_x, y+2*lh), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), thickness)
        else:
            cv2.putText(frame, f"DE XUAT: {calculate_optimal_green_time(i)}s", (start_x, y+2*lh), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,255), thickness)
    for i in range(max_directions):
        if traffic_light_status[i] == "GREEN":
            cv2.putText(frame, f"HUONG {chr(65+i)} DANG XANH", (900, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    return frame

# ------------------------ Khởi động hệ thống ------------------------
cap = cv2.VideoCapture('Bellevue_116th_NE12th__2017-09-11_07-08-32.mp4')
model = YOLO('yolov8n.pt')
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", mouseClick)
classnames = open('classes.txt').read().splitlines()
tracker = Sort(max_age=20)

# ------------------------ Vòng lặp chính ------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        cap = cv2.VideoCapture('Bellevue_116th_NE12th__2017-09-11_07-08-32.mp4')
        continue
    now = time.time()
    simulate_traffic_lights(now)

    # Vẽ các điểm và line/zone
    for point in click_points:
        cv2.circle(frame, point, 5, (255, 0, 0), -1)
    for i, line in enumerate(lines):
        color = (0,255,0) if traffic_light_status[i] == "GREEN" else (0,0,255)
        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), color, 3)
    for zones in queue_zones:
        for zone in zones:
            cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]), (0,255,255), 2)

    detections = np.empty((0, 5))
    result = model(frame, stream=True)
    for info in result:
        for box in info.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf, cls = int(box.conf[0]*100), int(box.cls[0])
            if classnames[cls] in ['car','bus','truck','motor'] and conf > 40:
                detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

    track_result = tracker.update(detections)
    for res in track_result:
        x1,y1,x2,y2,id = map(int, res)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.circle(frame, (cx, cy), 6, (0,0,255), -1)
        cvzone.putTextRect(frame, f'ID {id}', [x1, y1-10], scale=0.7)

        for i, line in enumerate(lines):
            if id in vehicle_positions[i]:
                px, py = vehicle_positions[i][id]
                speed = math.sqrt((cx - px)**2 + (cy - py)**2) * fps
                vehicle_speeds[i][id] = speed
                for zone in queue_zones[i]:
                    if check_vehicle_in_queue_zone(cx, cy, zone) and speed < queue_threshold_speed:
                        if id not in queue_counters[i]: queue_counters[i].append(id)
            vehicle_positions[i][id] = (cx, cy)
            if line[0] <= cx <= line[2] and abs(cy - line[1]) <= 20:
                if id not in counters[i]:
                    counters[i].append(id)
                    if vehicle_times[i]:
                        hw = now - max(vehicle_times[i].values())
                        headways[i].append(hw)
                    vehicle_times[i][id] = now

    frame = display_traffic_info(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
