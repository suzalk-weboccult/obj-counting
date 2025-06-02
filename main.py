
#TODO Add proper commenting and doc string 
#TODO: make it in class and object why it is only in one file 
#TODO: Code  seems to be generated from the AI , there are instrudction for not to use the AI 
#TODO : why not saving the output video? it is needed for verification of the counting and debugging the logic 

import cv2
import numpy as np
import time
import os

# Input video
# video_path = "/home/wot-suzal/internship/obj_counting/counting_diff_obj/bottle_test.mp4"
video_path="/home/wot-suzal/internship/obj_counting/counting_diff_obj/bottle_test.mp4"
# video_path="/home/wot-suzal/internship/obj_counting/counting_diff_obj/packet_counting_video3.mp4"
cap = cv2.VideoCapture(video_path)

ret, first_frame = cap.read()
if not ret:
    print("Failed to read video.")
    cap.release()
    exit()

# Resize and get dimensions
first_frame = cv2.resize(first_frame, (540, 380))
frame_height, frame_width = first_frame.shape[:2]
half_line_y = frame_height // 2

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Initialize counters
processed_frames = 0
object_count = 0

# Create output directory if not exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = os.path.join(output_dir, "inference_output.mp4")
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

# Non-max suppression function
def non_max_suppression_fast(boxes, overlapThresh=0.7):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    bboxes = boxes[:, :4].astype(int).tolist()
    scores = boxes[:, 4].astype(float).tolist()

    indices = cv2.dnn.NMSBoxes(bboxes, scores, score_threshold=0.0, nms_threshold=overlapThresh)

    if isinstance(indices, tuple) or isinstance(indices, np.ndarray):
        indices = np.array(indices).flatten().tolist()
    elif isinstance(indices, list):
        indices = [i[0] if isinstance(i, (list, tuple)) else i for i in indices]
    else:
        indices = []

    return [boxes[i] for i in indices]

# Start processing
start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (540, 380))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame, (0, half_line_y), (frame_width, half_line_y), (0, 255, 255), 2)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([x, y, x + w, y + h, area])

    filtered_boxes = non_max_suppression_fast(boxes)

    for box in filtered_boxes:
        x1, y1, x2, y2, _ = box
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

        center_x = x + w // 2
        center_y = y + h // 2

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)

        if half_line_y - 2 < center_y < half_line_y + 5:
            object_count += 1

    processed_frames += 1
    processing_time = time.time() - start_time
    fps = (processed_frames / processing_time) if processing_time > 0 else 0

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Count: {object_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Save the processed frame
    out.write(frame)

    # Show live frames
    cv2.imshow('Threshold', thresh)
    cv2.imshow('Frame', frame)

    prev_gray = gray.copy()

    if cv2.waitKey(7) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print("Total objects crossed the line:", object_count)
print("Inference video saved to:", output_path)
