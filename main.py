
#TODO Add proper commenting and doc string 
#TODO: make it in class and object why it is only in one file 
#TODO: Code  seems to be generated from the AI , there are instrudction for not to use the AI 
#TODO : why not saving the output video? it is needed for verification of the counting and debugging the logic 

import cv2
import numpy as np
import time
import os
from utils.utils import non_max_suppression_fast


class ObjectCounter:
    """
    Class to handle object detection and counting in a video stream based on motion detection.

    Attributes:
        video_path (str): Path to the input video.
        output_dir (str): Directory to save the output video.
        object_count (int): Count of objects crossing the mid-line.
        processed_frames (int): Total frames processed.
        frame_width (int): Width of each frame.
        frame_height (int): Height of each frame.
        half_line_y (int): Y-coordinate of the line used to count crossings.
        output_path (str): Path to the saved output video.
    """

    def __init__(self, video_path: str, output_dir: str = "output"):
        self.video_path = video_path
        self.output_dir = output_dir
        self.object_count = 0
        self.processed_frames = 0
        self.frame_width = 0
        self.frame_height = 0
        self.half_line_y = 0
        self.output_path = ""

        os.makedirs(output_dir, exist_ok=True)
        self.cap = cv2.VideoCapture(self.video_path)

        ret, first_frame = self.cap.read()
        if not ret:
            raise ValueError("Failed to read video.")

        self.first_frame = cv2.resize(first_frame, (540, 380))
        self.prev_gray = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2GRAY)

        self.frame_height, self.frame_width = self.first_frame.shape[:2]
        self.half_line_y = self.frame_height // 2

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_path = os.path.join(self.output_dir, "inference_output.mp4")
        self.out_writer = cv2.VideoWriter(self.output_path, fourcc, 20.0, (self.frame_width, self.frame_height))

    def process_video(self):
        """
        Processes the video frame by frame, applies motion-based object detection,
        draws bounding boxes, and counts objects crossing a horizontal line.
        """
        start_time = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (540, 380))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Compute frame difference
            diff = cv2.absdiff(self.prev_gray, gray)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

            # Dilate to fill gaps
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=2)

            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.line(frame, (0, self.half_line_y), (self.frame_width, self.half_line_y), (0, 255, 255), 2)

            boxes = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append([x, y, x + w, y + h, area])

            # Apply Non-Maximum Suppression
            filtered_boxes = non_max_suppression_fast(boxes)

            for box in filtered_boxes:
                x1, y1, x2, y2, _ = box
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

                center_x = x + w // 2
                center_y = y + h // 2

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)

                # Check if object crosses the line
                if self.half_line_y - 2 < center_y < self.half_line_y + 5:
                    self.object_count += 1

            self.processed_frames += 1
            processing_time = time.time() - start_time
            fps = (self.processed_frames / processing_time) if processing_time > 0 else 0

            # Display metrics
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f'Count: {self.object_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Write and show frames
            self.out_writer.write(frame)
            cv2.imshow('Threshold', thresh)
            cv2.imshow('Frame', frame)

            self.prev_gray = gray.copy()

            if cv2.waitKey(6) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        """Releases resources and prints the final results."""
        self.cap.release()
        self.out_writer.release()
        cv2.destroyAllWindows()
        print("Total objects crossed the line:", self.object_count)
        print("Inference video saved to:", self.output_path)


if __name__ == "__main__":
    # Example usage
    # video_path = "/home/wot-suzal/internship/obj_counting/counting_diff_obj/bottle_test.mp4"
    # video_path="/home/wot-suzal/internship/obj_counting/counting_diff_obj/packet_counting_video3.mp4"
    video_path="/home/wot-suzal/internship/obj_counting/counting_diff_obj/packet-counting.mp4"

    counter = ObjectCounter(video_path)
    counter.process_video()
