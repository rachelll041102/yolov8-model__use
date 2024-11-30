import numpy as np
import cv2
# YOLOv8
from ultralytics import YOLO

model = YOLO('yolov8n.pt')


rtsp_url = 'rtsp://192.168.1.108/user=admin&password=123456/id=1&type=0'
cap = cv2.VideoCapture(rtsp_url)
names = model.names			# model的名字的键值对 例子： 0:person 62:tv
frameCount=0
while cap.isOpened():
    # 捕获帧
    ret, frame = cap.read()
    frameCount += 1
    if not ret:
        print("Error reading frame")
        break
        
    # 20帧检测一次
    if frameCount % 20 != 0:
        continue
    if frameCount == 20:
        frameCount = 0
        
      
    # cls: tensor([ 0., 62., 62.,  0.,  0.])
    # 使用YOLOv8n.pt进行物体检测
    results = model.predict(frame)
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        cls = result.boxes.cls		    # 显示检测到的物体的名字，例子： cls: tensor([ 0., 62., 62.,  0.,  0.])
        print(cls)
        for box, conf,c in zip(boxes, confidences,cls):		# 方框坐标 与名字 的集合进行绑定
            x1, y1, x2, y2 = map(int, box[:4])			
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
            frame = cv2.putText(frame, names[int(c)], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
    cv2.imshow("frame",frame)
         # 按 'q' 键退出
    if cv2.waitKey(1) == ord('q'):
        break

# 释放相机和关闭所有 OpenCV 窗口
cap.release()
cv2.destroyAllWindows()
