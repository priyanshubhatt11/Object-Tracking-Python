import cv2
from tracker import *

#create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("highway3.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape

    #extract region of interest
    roi = frame[200: 720, 400: 920] # left, right, up, down

    #1 Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    countours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #find boundaries (img, retrieval mode, approximation method)
    detections = []

    for cnt in countours:

        #calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 250:
            #cv2.drawContours(roi, [cnt], -1, (0,255,0),2)
            x, y, w, h = cv2.boundingRect(cnt)
            
            detections.append([x,y,w,h])


    #2 Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x,y -15), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
        cv2.rectangle(roi, (x, y), (x+w, y+h), (0,255,0), 3)

    print(detections)
    cv2.imshow("roi", roi)
    cv2.imshow("Mask", mask)    # black screen, more white means object

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(10)
    

cap.release()
cv2.destroyAllWindows()