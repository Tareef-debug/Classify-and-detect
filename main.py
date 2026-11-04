import cv2
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
camera = cv2.VideoCapture(0)
confidence = 0.5
while True:
    success,frame = camera.read()
    if not success:
        break
    results = model(frame, verbose=False)
    for result in results:
        for box in result.boxes:
            score=float(box.conf[0])
            class_id=int(box.cls[0])
            name=model.names[class_id]
            if score>=confidence:
                x1,y1,x2,y2=map(int,box.xyxy[0])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,234,0),2)
                label=f"{name}:{score:.1f}"
                cv2.putText(frame,label,(x1,max(20,y1-10)),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,0.6,(0,255,0),2)
    cv2.imshow("Live object detector",frame)
    key= cv2.waitKey(1)&0xFF
    if key==ord('s'):
        cv2.imwrite("screenshot.jpg",frame)
        print("Screenshot saved successfully!")
    elif key == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()