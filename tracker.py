import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime, timedelta
import csv
import time

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model/detect.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load label map
labels = []
with open("labelmap.txt", "r") as f:
    for line in f:
        if "display_name" in line:
            labels.append(line.strip().split(' ')[-1].replace('"', ''))

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for default camera

# Parameters for counting
maxDisappeared = 50
maxDistance = 50
minTrackingDuration = 1.0  # Minimum tracking duration in seconds
entry_buffer_time = 2.0  # Time buffer to prevent re-entry of the same object in seconds

# Placeholder for frame dimensions
frame_width = None
frame_height = None

class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50, minTrackingDuration=1.0, entry_buffer_time=2.0):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
        self.minTrackingDuration = minTrackingDuration
        self.entry_buffer_time = entry_buffer_time
        self.entry_times = {}
        self.exit_times = {}
        self.records = []
        self.last_registered_time = {}

    def register(self, centroid):
        current_time = datetime.now()
        for objectID, last_time in self.last_registered_time.items():
            if (current_time - last_time).total_seconds() < self.entry_buffer_time:
                if np.linalg.norm(np.array(centroid) - np.array(self.objects[objectID])) < self.maxDistance:
                    return None
        
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.entry_times[self.nextObjectID] = current_time
        self.last_registered_time[self.nextObjectID] = current_time
        objectID = self.nextObjectID
        self.nextObjectID += 1
        return objectID

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        exit_time = datetime.now()
        if objectID in self.entry_times:
            entry_time = self.entry_times[objectID]
            if (exit_time - entry_time).total_seconds() >= self.minTrackingDuration:
                self.records.append((objectID, entry_time, exit_time))
            del self.entry_times[objectID]
        self.exit_times[objectID] = exit_time

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - inputCentroids[np.newaxis, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                if D[row, col] > self.maxDistance:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

def remove_duplicate_records(records, time_threshold=1.0):
    filtered_records = []
    for i in range(len(records)):
        if i == 0 or (records[i][2] - records[i-1][2]).total_seconds() > time_threshold:
            filtered_records.append(records[i])
    return filtered_records

def print_records(records, csv_writer):
    filtered_records = remove_duplicate_records(records)
    for record in filtered_records:
        person_id, entry_time, exit_time = record
        print(f"Person {person_id} entered at {entry_time.strftime('%Y-%m-%d %H:%M:%S')} and left at {exit_time.strftime('%Y-%m-%d %H:%M:%S')}")
        csv_writer.writerow([f"Person {person_id}", entry_time.strftime('%Y-%m-%d %H:%M:%S'), exit_time.strftime('%Y-%m-%d %H:%M:%S')])

ct = CentroidTracker(maxDisappeared=maxDisappeared, maxDistance=maxDistance, minTrackingDuration=minTrackingDuration, entry_buffer_time=entry_buffer_time)

# Open CSV file for writing
csv_file = open('tracking_records.csv', mode='w')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Person ID', 'Entry Time', 'Exit Time'])

start_time = datetime.now()
end_time = start_time + timedelta(minutes=1)  # Run for 1 minute

while datetime.now() < end_time:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_width is None or frame_height is None:
        frame_height, frame_width = frame.shape[:2]

    # Preprocess the frame
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_resized = img.resize((300, 300))
    input_data = np.expand_dims(img_resized, axis=0).astype(np.uint8)

    # Perform the detection
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    rects = []
    for i in range(len(scores)):
        if scores[i] > 0.5 and int(classes[i]) < len(labels) and labels[int(classes[i])] == 'person':  # Adjust threshold and detect only people
            ymin, xmin, ymax, xmax = boxes[i]
            (left, top, right, bottom) = (xmin * frame.shape[1], ymin * frame.shape[0],
                                          xmax * frame.shape[1], ymax * frame.shape[0])
            rects.append((int(left), int(top), int(right), int(bottom)))
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)  # Draw bounding box

    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():
        text = f"ID {objectID}"
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    print_records(ct.records, csv_writer)
    ct.records.clear()

    # Display the frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print remaining records after the loop ends
print_records(ct.records, csv_writer)

cap.release()
cv2.destroyAllWindows()
csv_file.close()
