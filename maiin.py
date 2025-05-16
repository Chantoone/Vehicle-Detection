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
max_directions = 2

counters = [[] for _ in range(max_directions)]
vehicle_times = [{} for _ in range(max_directions)]
vehicle_speeds = [{} for _ in range(max_directions)]
vehicle_positions = [{} for _ in range(max_directions)]
headways = [[] for _ in range(max_directions)]
queue_counters = [[] for _ in range(max_directions)]
last_red_time = [0 for _ in range(max_directions)]
estimated_green = [10 for _ in range(max_directions)]

min_green_time = [15, 15]
max_green_time = [60, 60]
traffic_light_status = ["RED", "RED"]
green_start_time = [0, 0]
current_cycle_start = time.time()
total_cycle_time = 120  # Tổng thời gian chu kỳ đèn (giây)

fps = 30
queue_threshold_speed = 3
headway_threshold = 3
confidence_threshold = 30  # Giảm ngưỡng tin cậy từ 40 xuống 30


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
            direction_index = (len(zones_draw) // 2 - 1) % max_directions  # Tính index cho hướng hiện tại
            zone = (zones_draw[-2][0], zones_draw[-2][1], zones_draw[-1][0], zones_draw[-1][1])

            # Đảm bảo queue_zones đã được khởi tạo đúng cách
            while len(queue_zones) <= direction_index:
                queue_zones.append([])

            queue_zones[direction_index].append(zone)
            print(f"Added zone to direction {direction_index}: {zone}")
            print("Queue Zones:", queue_zones)


def check_vehicle_in_queue_zone(bbox, zone):
    x1, y1, x2, y2 = bbox
    zone_x1, zone_y1, zone_x2, zone_y2 = zone

    # Đảm bảo zone_x1 < zone_x2 và zone_y1 < zone_y2
    if zone_x1 > zone_x2:
        zone_x1, zone_x2 = zone_x2, zone_x1
    if zone_y1 > zone_y2:
        zone_y1, zone_y2 = zone_y2, zone_y1

    # Kiểm tra nếu bất kỳ góc hoặc tâm xe nằm trong zone
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    return (
            (zone_x1 <= x1 <= zone_x2 and zone_y1 <= y1 <= zone_y2) or
            (zone_x1 <= x2 <= zone_x2 and zone_y1 <= y1 <= zone_y2) or
            (zone_x1 <= x1 <= zone_x2 and zone_y1 <= y2 <= zone_y2) or
            (zone_x1 <= x2 <= zone_x2 and zone_y1 <= y2 <= zone_y2) or
            (zone_x1 <= cx <= zone_x2 and zone_y1 <= cy <= zone_y2)
    )


def calculate_optimal_green_time(i):
    # Cải thiện tính toán thời gian đèn xanh tối ưu
    q_len = len(queue_counters[i])

    # Tính thời gian headway trung bình (nếu không có dữ liệu thì sử dụng giá trị mặc định)
    avg_hw = sum(headways[i]) / len(headways[i]) if headways[i] else 2.5

    # Tính thời gian cơ bản dựa trên độ dài hàng đợi và headway trung bình
    base_green = max(min_green_time[i], q_len * avg_hw)

    # Tính hệ số theo lưu lượng (số lượng xe đã đi qua)
    flow_count = len(counters[i])
    vol_factor = max(1.0, min(2.0, flow_count / 8))

    # Tính hệ số dựa trên tốc độ trung bình
    speeds = list(vehicle_speeds[i].values())
    avg_spd = sum(speeds) / len(speeds) if speeds else 0

    # Điều chỉnh hệ số tốc độ cho 2 hướng: tốc độ càng chậm, thời gian xanh càng dài
    if avg_spd < 5:  # Rất chậm hoặc tắc nghẽn
        spd_factor = 1.8
    elif avg_spd < 15:  # Chậm
        spd_factor = 1.4
    elif avg_spd < 30:  # Trung bình
        spd_factor = 1.0
    else:  # Nhanh
        spd_factor = 0.8

    # Thời gian chờ: thời gian đã trôi qua kể từ lần cuối cùng đèn chuyển sang đỏ
    wait_time = time.time() - last_red_time[i]
    wait_factor = min(1.5, max(1.0, wait_time / 60))  # Hệ số từ 1.0-1.5

    # Tính toán thời gian xanh tổng hợp
    green_time = base_green * vol_factor * spd_factor * wait_factor

    # Giới hạn trong khoảng min-max
    return int(max(min_green_time[i], min(max_green_time[i], green_time)))


def calculate_direction_priority():
    priorities = []
    now = time.time()

    for i in range(max_directions):
        # Số lượng xe trong hàng đợi (tăng mức ưu tiên cho hệ thống 2 hướng)
        queue_length = len(queue_counters[i])
        queue_factor = queue_length * 3  # Tăng trọng số của hàng đợi

        # Thời gian chờ (thời gian từ khi chuyển sang đỏ)
        wait_time = now - last_red_time[i] if last_red_time[i] > 0 else 0
        wait_factor = min(5.0, wait_time / 25)  # Tăng hệ số chờ tối đa và giảm thời gian cần thiết

        # Tốc độ trung bình (nếu có)
        speeds = list(vehicle_speeds[i].values())
        avg_speed = sum(speeds) / len(speeds) if speeds else 0

        # Hệ số tốc độ: ưu tiên hướng có xe di chuyển chậm hơn
        speed_factor = 4.0 if avg_speed < 5 else (2.5 if avg_speed < 15 else 1.0)

        # Công thức ưu tiên tổng hợp cho hệ thống 2 hướng
        priority_score = queue_factor + wait_factor * 15 + speed_factor * 5

        # Thêm vào danh sách
        priorities.append((i, priority_score))

    # Sắp xếp theo độ ưu tiên giảm dần
    return sorted(priorities, key=lambda x: x[1], reverse=True)


def simulate_traffic_lights(now):
    global traffic_light_status, green_start_time, estimated_green, last_red_time, current_cycle_start

    # Kiểm tra nếu cần bắt đầu chu kỳ mới
    if now - current_cycle_start > total_cycle_time:
        current_cycle_start = now
        # Reset các trạng thái nếu cần

    # Tìm hướng đang có đèn xanh (nếu có)
    current = -1
    for i in range(max_directions):
        if traffic_light_status[i] == "GREEN":
            current = i
            green_duration = now - green_start_time[i]

            # Kiểm tra thời gian đèn xanh có vượt quá thời gian ước tính
            if green_duration >= estimated_green[i]:
                # Chuyển sang đèn đỏ
                traffic_light_status[i] = "RED"
                last_red_time[i] = now
                # Reset bộ đếm dòng xe khi đèn chuyển đỏ
                counters[i] = []
                current = -1
                print(f"Chuyển đèn XANH sang ĐỎ cho hướng {i} sau {green_duration:.1f}s")

    # Nếu không có hướng nào đang xanh, chọn hướng tiếp theo
    if current == -1:
        # Tính toán ưu tiên cho các hướng và chọn hướng tiếp theo
        priorities = calculate_direction_priority()

        # Vì chỉ có 2 hướng, chúng ta có thể đơn giản hóa logic
        # Nếu cả hai hướng có ưu tiên bằng nhau, chọn hướng có thời gian chờ lâu hơn
        if len(priorities) >= 2 and abs(priorities[0][1] - priorities[1][1]) < 5:
            # Nếu ưu tiên gần bằng nhau, chọn hướng đã chờ lâu hơn
            wait_times = [time.time() - last_red_time[i] for i in range(max_directions)]
            next_i = 0 if wait_times[0] > wait_times[1] else 1
        else:
            next_i = priorities[0][0]

        # Chuyển sang đèn xanh cho hướng được chọn
        traffic_light_status[next_i] = "GREEN"
        green_start_time[next_i] = now

        # Tính thời gian đèn xanh tối ưu cho hướng này
        estimated_green[next_i] = calculate_optimal_green_time(next_i)

        # Reset hàng đợi cho hướng này
        queue_counters[next_i] = []

        print(f"Chuyển đèn ĐỎ sang XANH cho hướng {next_i}, dự kiến: {estimated_green[next_i]}s")


def display_traffic_info(frame):
    font_scale, thickness, lh = 0.6, 1, 25
    start_x, start_y = 10, 30
    panel_w, panel_h = 250, max_directions * lh * 3 + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (start_x - 5, start_y - 25), (start_x + panel_w, start_y + panel_h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    for i in range(max_directions):
        y = start_y + i * lh * 3
        # Số lượng xe và hàng đợi
        cv2.putText(frame, f"Huong {chr(65 + i)}: {len(counters[i])} xe | Hang doi: {len(queue_counters[i])}",
                    (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # Tốc độ trung bình
        speeds = list(vehicle_speeds[i].values())
        spd = sum(speeds) / len(speeds) if speeds else 0
        cv2.putText(frame, f"Speed TB: {spd:.1f}px/s", (start_x, y + lh), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness)

        # Trạng thái đèn và thời gian
        if traffic_light_status[i] == "GREEN":
            remaining = estimated_green[i] - (time.time() - green_start_time[i])
            cv2.putText(frame, f"DEN XANH: {int(remaining)}s", (start_x, y + 2 * lh), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 255, 0), thickness)
        else:
            wait_time = time.time() - last_red_time[i] if last_red_time[i] > 0 else 0
            cv2.putText(frame, f"CHO: {int(wait_time)}s | DE XUAT: {calculate_optimal_green_time(i)}s",
                        (start_x, y + 2 * lh), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

    # Hiển thị hướng đang xanh
    for i in range(max_directions):
        if traffic_light_status[i] == "GREEN":
            cv2.putText(frame, f"HUONG {chr(65 + i)} DANG XANH", (900, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                        2)

    # Thêm thông tin debugging
    cv2.putText(frame, f"Threshold: {confidence_threshold}%", (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0),
                2)
    return frame


# ------------------------ Khởi động hệ thống ------------------------
cap = cv2.VideoCapture('Bellevue_116th_NE12th__2017-09-11_07-08-32.mp4')
model = YOLO('yolov8n.pt')
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", mouseClick)
classnames = open('classes.txt').read().splitlines()
tracker = Sort(max_age=20)

# Khởi tạo queue_zones cho mỗi hướng
queue_zones = [[] for _ in range(max_directions)]

# ------------------------ Vòng lặp chính ------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        cap = cv2.VideoCapture('Bellevue_116th_NE12th__2017-09-11_07-08-32.mp4')
        continue

    # Tạo một bản sao của khung hình để xử lý
    frame_processed = frame.copy()

    now = time.time()
    simulate_traffic_lights(now)

    # Vẽ các điểm và line/zone
    for point in click_points:
        cv2.circle(frame, point, 5, (255, 0, 0), -1)
    for i, line in enumerate(lines):
        color = (0, 255, 0) if traffic_light_status[i] == "GREEN" else (0, 0, 255)
        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), color, 3)

    # Vẽ các zone
    for i, zones in enumerate(queue_zones):
        if i < len(queue_zones):
            for zone in zones:
                color = (0, 255, 255) if traffic_light_status[i] == "GREEN" else (0, 200, 200)
                cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]), color, 2)

    detections = np.empty((0, 5))
    result = model(frame, stream=True)

    # Count detected objects for debugging
    detected_count = 0

    for info in result:
        for box in info.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf, cls = float(box.conf[0]) * 100, int(box.cls[0])

            if cls < len(classnames) and classnames[cls] in ['car', 'bus', 'truck',
                                                             'motor'] and conf > confidence_threshold:
                detected_count += 1
                detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

    track_result = tracker.update(detections)

    # Hiển thị số lượng phát hiện cho mục đích gỡ lỗi
    cv2.putText(frame, f"Detected: {detected_count} vehicles", (10, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0),
                2)

    for res in track_result:
        x1, y1, x2, y2, id = map(int, res)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
        cvzone.putTextRect(frame, f'ID {id}', [x1, y1 - 10], scale=0.7)

        for i, line in enumerate(lines):
            if i < len(lines):  # Đảm bảo không vượt quá số lines đã tạo
                # Kiểm tra vị trí xe và tính toán tốc độ
                if id in vehicle_positions[i]:
                    px, py = vehicle_positions[i][id]
                    distance = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)

                    # Tính toán tốc độ và áp dụng bộ lọc trung bình động
                    current_speed = distance * fps  # pixels/second
                    if id in vehicle_speeds[i]:
                        # Áp dụng bộ lọc trung bình động để làm mượt tốc độ
                        vehicle_speeds[i][id] = 0.7 * vehicle_speeds[i][id] + 0.3 * current_speed
                    else:
                        vehicle_speeds[i][id] = current_speed

                    # Kiểm tra xe trong các zone hàng đợi - chỉ thêm mỗi ID một lần
                    if i < len(queue_zones):
                        in_queue = False
                        for zone in queue_zones[i]:
                            if check_vehicle_in_queue_zone((x1, y1, x2, y2), zone) and vehicle_speeds[i][
                                id] < queue_threshold_speed:
                                in_queue = True
                                break

                        if in_queue and id not in queue_counters[i]:
                            queue_counters[i].append(id)

                # Lưu vị trí hiện tại của xe
                vehicle_positions[i][id] = (cx, cy)

                # Kiểm tra xe đi qua line
                if i < len(lines):  # Chắc chắn chỉ số i hợp lệ
                    x1_line, y1_line, x2_line, y2_line = line

                    # Tính toán hệ số góc và hằng số của đường thẳng
                    if x2_line != x1_line:  # Tránh chia cho 0
                        m = (y2_line - y1_line) / (x2_line - x1_line)
                        b = y1_line - m * x1_line

                        # Tính điểm gần nhất từ xe đến đường thẳng
                        if abs(m) > 0.001:  # Nếu đường thẳng không nằm ngang
                            x_nearest = (cx + m * cy - m * b) / (m ** 2 + 1)
                            y_nearest = m * x_nearest + b
                        else:  # Nếu đường thẳng gần như nằm ngang
                            x_nearest = cx
                            y_nearest = y1_line

                        # Tính khoảng cách từ xe đến đường thẳng
                        distance_to_line = math.sqrt((cx - x_nearest) ** 2 + (cy - y_nearest) ** 2)

                        # Kiểm tra nếu điểm gần nhất nằm trên đoạn thẳng
                        is_on_segment = min(x1_line, x2_line) <= x_nearest <= max(x1_line, x2_line) and \
                                        min(y1_line, y2_line) <= y_nearest <= max(y1_line, y2_line)

                        # Nếu xe đủ gần đoạn thẳng, đánh dấu là đã vượt qua line
                        if distance_to_line < 25 and is_on_segment:
                            if id not in counters[i]:
                                counters[i].append(id)
                                # Tính toán headway
                                if vehicle_times[i]:
                                    latest_vehicle_time = max(vehicle_times[i].values())
                                    hw = now - latest_vehicle_time
                                    # Chỉ thêm headway hợp lệ (không quá lớn)
                                    if hw < headway_threshold * 5:  # Giới hạn headway tối đa
                                        headways[i].append(hw)
                                        # Giữ kích thước danh sách headways hợp lý
                                        if len(headways[i]) > 20:
                                            headways[i] = headways[i][-20:]

                                # Lưu thời gian xe đi qua
                                vehicle_times[i][id] = now

    frame = display_traffic_info(frame)
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key để thoát
        break
    elif key == ord('+'):  # Phím + để tăng ngưỡng tin cậy
        confidence_threshold = min(confidence_threshold + 5, 100)
    elif key == ord('-'):  # Phím - để giảm ngưỡng tin cậy
        confidence_threshold = max(confidence_threshold - 5, 5)

cap.release()
cv2.destroyAllWindows()