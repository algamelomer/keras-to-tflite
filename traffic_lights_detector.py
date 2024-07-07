import cv2
import numpy as np

classnames = []
classfile = 'files/thing.names'

with open(classfile, 'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')

# Load the model and configuration
p = 'files/frozen_inference_graph.pb'
v = 'files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

net = cv2.dnn_DetectionModel(p, v)
net.setInputSize(320, 230)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize video capture (0 for webcam, or filename for video file)
# source = 'http://192.168.59.152:5000/video_feed'
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

def apply_roi_mask(frame, box):
    mask = np.zeros_like(frame)
    x, y, w, h = box
    mask[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
    return mask

def detect_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define color ranges for red and green
    red_lower1 = np.array([0, 70, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 70, 50])
    red_upper2 = np.array([180, 255, 255])
    green_lower = np.array([40, 70, 50])
    green_upper = np.array([90, 255, 255])

    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)


    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    red_mask = red_mask1 | red_mask2

    red_count = cv2.countNonZero(red_mask)
    green_count = cv2.countNonZero(green_mask)

    if red_count > green_count:
        return 'Red'
    elif green_count > red_count:
        return 'Green'
    else:
        return 'Unknown'

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, -1)
    if not ret:
        break

    # Perform object detection
    classIds, confs, bbox = net.detect(frame, confThreshold=0.3)

    if len(classIds) > 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            label = classnames[classId - 1]
            if label == 'traffic light':  # Check if the detected object is a traffic light
                x, y, w, h = box
                roi = frame[y:y+h, x:x+w]
                color = detect_color(roi)
                cv2.putText(frame, color,
                            (x + 10, y + h + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), thickness=2)

            cv2.rectangle(frame, box, color=(255, 0, 0), thickness=3)
            cv2.putText(frame, label,
                        (box[0] + 10, box[1] + 20),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), thickness=2)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
