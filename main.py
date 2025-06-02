import cv2
import numpy as np
import time


# cap = cv2.VideoCapture('/home/wot-suzal/internship/obj_counting/counting_diff_obj/packet-counting.mp4')
# cap=cv2.VideoCapture("/home/wot-suzal/internship/obj_counting/counting_diff_obj/bottle_test.mp4")
cap=cv2.VideoCapture("/home/wot-suzal/internship/obj_counting/counting_diff_obj/packet_counting_video3.mp4")

ret, first_frame = cap.read()
if not ret:
    print("Failed to read video.")
    cap.release()
    exit()

first_frame = cv2.resize(first_frame, (540, 380))
frame_height, frame_width = first_frame.shape[:2]
half_line_y = frame_height // 2  # Horizontal line at half height

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

processed_frames=0
# Object counter
object_count = 0
# To store previous centers and avoid double counting
# detected_centers = []

# Apply NMS
def non_max_suppression_fast(boxes, overlapThresh=0.7):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    bboxes = boxes[:, :4].astype(int).tolist()
    scores = boxes[:, 4].astype(float).tolist()

    indices = cv2.dnn.NMSBoxes(
        bboxes=bboxes, 
        scores=scores, 
        score_threshold=0.0, 
        nms_threshold=overlapThresh
    )

    # Flatten indices safely
    if isinstance(indices, tuple) or isinstance(indices, np.ndarray):
        indices = np.array(indices).flatten().tolist()
    elif isinstance(indices, list):
        indices = [i[0] if isinstance(i, (list, tuple)) else i for i in indices]
    else:
        indices = []

    return [boxes[i] for i in indices]

start_time = time.time()    
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (540, 380))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Frame difference
    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the halfway line
    cv2.line(frame, (0, half_line_y), (frame_width, half_line_y), (0, 255, 255), 2)

    new_centers = []

    # Collect bounding boxes and areas
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([x, y, x + w, y + h, area])  # [x1, y1, x2, y2, score]

    filtered_boxes = non_max_suppression_fast(boxes)

    # Now loop through the filtered boxes
    for box in filtered_boxes:
        x1, y1, x2, y2, _ = box
        x1, y1, x2, y2, _ = box
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

        center_x = x + w // 2
        center_y = y + h // 2

        # Draw bounding box and center
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)
        new_centers.append((center_x, center_y))

        # If the center crossed the half-line and not already counted
        if center_y > half_line_y - 2 and center_y < half_line_y + 5:
            # if not any(abs(center_x - cx) < 10 and abs(center_y - cy) < 10 for cx, cy in detected_centers):
                object_count += 1
                # detected_centers.append((center_x, center_y))
    processed_frames+=1
    processing_time = time.time() - start_time
    fps = processed_frames / processing_time if processing_time > 0 else 0

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display count on frame
    cv2.putText(frame, f'Count: {object_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show frames
    cv2.imshow('Threshold', thresh)
    cv2.imshow('Frame', frame)

    prev_gray = gray.copy()  # Update previous frame

    # Exit on pressing 'q'
    if cv2.waitKey(11) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("Total objects crossed the line:", object_count)
