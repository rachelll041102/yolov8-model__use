import numpy as np
import cv2
# YOLOv8
from ultralytics import YOLO


# 定义连接关系（每对关键点的索引）
connections = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 鼻子与眼睛和耳朵
    (5, 6), (5, 7), (6, 8),          # 肩膀与肘部
    (7, 9), (8, 10),                 # 左右手肘与手腕
    (5, 11), (6, 12),                # 肩膀与髋部
    (11, 13), (12, 14),              # 髋部与膝部
    (13, 15), (14, 16)               # 膝部与脚踝
]


def draw_pose(image, keypoints, confidence_threshold=0.0):
    for i in range(17):
        x, y = keypoints[i]
        if x==0 and y==0:
            continue
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # 绘制关键点

    for start, end in connections:
    	
        start_point = (int(keypoints[start][0]), int(keypoints[start][1]))
        end_point = (int(keypoints[end][0]), int(keypoints[end][1]))

        if (start_point == (0,0)) or (end_point == (0,0)):
            continue
        # 举手 判断手肘和手腕
        if ( (start == 7 and end == 9) or ( start == 8 and end == 10) ) and ( start_point[1]>end_point[1]):
            print(start_point)
            print(end_point)
            print("raise hand")
            print()
            cv2.line(image, start_point, end_point, (0,0, 255), 2)  # 绘制连接线
            cv2.putText(image,"raise hand",start_point,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        # 蹲下 判断髋部和膝部    
        elif ( (start == 11 and end == 13) or ( start == 12 and end == 14) ) and ( start_point[1]>end_point[1]):
            print(start_point)
            print(end_point)
            print("dun  xia")
            print()
            cv2.line(image, start_point, end_point, (0,0, 255), 2)  # 绘制连接线
            cv2.putText(image,"raise hand",start_point,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        else:
            cv2.line(image, start_point, end_point, (255, 0, 0), 2)  # 绘制连接线
 



# 加载预训练的YOLOv8模型
model = YOLO('yolov8n-pose.pt')
# RTSP 地址
rtsp_url = 'rtsp://192.168.1.108/user=admin&password=123456/id=1&type=0'
cap = cv2.VideoCapture(rtsp_url)

frameCount = 0

while cap.isOpened():
    # 捕获帧
    ret, frame = cap.read()
    
    frameCount += 1
    if not ret:
        print("Error reading frame")
        break

    if frameCount % 20 != 0:
        continue
    if frameCount == 20:
        frameCount = 0
    # 使用YOLOv8-pose.pt进行人体姿态检测
    results = model(frame)
    # 迭代检测到的人体姿态
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()                   
        keypoints = result.keypoints.cpu().numpy()  # 获取关键点
        box_index = 0
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box[:4])
            label = f'person {conf:.2f}'
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

            draw_pose(frame,keypoints[box_index].xy[0])

            frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 显示帧
    cv2.imshow('frame', frame)
    # 按 'q' 键退出
    if cv2.waitKey(1) == ord('q'):
        break

# 释放相机和关闭所有 OpenCV 窗口
cap.release()
cv2.destroyAllWindows()
