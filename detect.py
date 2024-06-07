import cv2
import torch
from yolov5 import detect  # Assuming YOLOv5 detect.py is in the same directory

# Load the YOLOv5 model (ensure you have the 'yolov5' directory in your project directory)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or 'yolov5m', 'yolov5l', 'yolov5x'

# Function to perform detection and tracking
def run_detection_and_tracking(video_source):
    # Initialize video capture
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    # Initialize an empty dictionary to hold the trackers for each object
    trackers = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv5 inference
        results = model(frame)

        # Extract bounding boxes and class labels
        bboxes = []
        for result in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2, conf, cls = result
            if conf > 0.5:  # Confidence threshold
                bboxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

        # Update tracker
        if not trackers:
            for bbox in bboxes:
                # Initialize a new tracker for each object
                tracker = cv2.TrackerMIL_create()
                tracker.init(frame, bbox)
                # Store the tracker in the dictionary with a unique identifier
                trackers[len(trackers)] = tracker
        else:
            for object_id, tracker in list(trackers.items()):
                success, bbox = tracker.update(frame)
                if success:
                    bboxes.append(tuple(map(int, bbox)))
                else:
                    del trackers[object_id]  # Remove tracker if tracking fails

        # Draw bounding boxes and labels on the frame
        for i, bbox in enumerate(bboxes):
            p1 = (bbox[0], bbox[1])
            p2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
            cv2.putText(frame, f"ID {i+1}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Object Detection and Tracking', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run detection and tracking on video source (0 for webcam or provide path to video file)
run_detection_and_tracking(0)
