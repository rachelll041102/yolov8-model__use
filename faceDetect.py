import cv2
from ultralytics import YOLO
import time


# 加载预训练的YOLOv8模型
model = YOLO('yolov8n-face-lindevs.pt')

# RTSP 地址
# rtsp_url = 'rtsp://192.168.1.109/user=admin&password=123456/id=1&type=0'
rtsp_url = 'rtsp://admin:admin@192.168.1.100:8554/live'
cap = cv2.VideoCapture(rtsp_url)

# 初始化帧计数器
frame_count = 0

# 创建一个可调节大小的窗口
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

# 用来跳帧统计
frameCount = 0

while cap.isOpened():



    # 捕获帧
    ret, frame = cap.read()
    frameCount+=1
    if not ret:
        print("Error reading frame")
        break
    # 10帧检测一下
    if frameCount % 20 != 0:
        continue
    if frameCount == 20:
        frameCount = 0
    
    # 使用YOLOv8进行人脸检测
    results = model(frame)

    # 迭代检测到的面部
    for result in results:
        # 获取检测框坐标和置信度
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box[:4])
            label = f'face {conf:.2f}'
            # 在帧上绘制矩形框
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # 添加标签
            frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    
    # 显示帧
    cv2.imshow('frame', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) == ord('q'):
        break

# 释放视频流
cap.release()
# 关闭所有 OpenCV 窗口
cv2.destroyAllWindows()
