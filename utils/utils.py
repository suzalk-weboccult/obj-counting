import numpy as np
import cv2

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