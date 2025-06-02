import numpy as np
import cv2 as cv

class ObjectCounter:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video = cv.VideoCapture(video_path)

        if not self.video.isOpened():
            raise IOError("Cannot open video file")

        self.bg_subtractor = cv.createBackgroundSubtractorKNN(
            history=500, detectShadows=False, dist2Threshold=700
        )

        self.count = 0
        self.flag = 0
        self.frame_size = (500, 500)

        # ROI bounds
        self.top, self.bottom = 170, 330
        self.left, self.right = 150, 300

    def preprocess(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)

        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((10, 10), np.uint8)

        mask_closed = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, kernel_close)
        mask_opened = cv.morphologyEx(mask_closed, cv.MORPH_OPEN, kernel_open)

        return mask_opened

    def draw_lines(self, frame):
        # Horizontal boundaries
        cv.line(frame, (0, self.top), (self.frame_size[0], self.top), (0, 255, 0), 2)
        cv.line(frame, (0, self.bottom), (self.frame_size[0], self.bottom), (0, 255, 0), 2)

        # Vertical boundaries
        cv.line(frame, (self.left, 0), (self.left, self.frame_size[1]), (255, 0, 0), 2)
        cv.line(frame, (self.right, 0), (self.right, self.frame_size[1]), (255, 0, 0), 2)

    def count_objects(self, mask, frame):
        blank_img = np.zeros((*self.frame_size, 3), dtype='uint8')
        gray_blank = cv.cvtColor(blank_img, cv.COLOR_BGR2GRAY)

        roi = mask[self.top:self.bottom, self.left:self.right]
        gray_blank[self.top:self.bottom, self.left:self.right] = roi

        contours, _ = cv.findContours(gray_blank, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            self.flag = 0
            return gray_blank

        temp = 0
        for contour in contours:
            rect = cv.minAreaRect(contour)
            box = cv.boxPoints(rect)
            box = np.int32(box)

            x_vals = box[:, 0]
            y_vals = box[:, 1]

            cv.drawContours(frame, [box], 0, (0, 0, 255), 2)

            if (np.all(y_vals >= self.top) and np.all(y_vals <= self.bottom)
                    and np.all(x_vals >= self.left) and np.all(x_vals <= self.right)
                    and cv.contourArea(contour) > 300):

                if self.flag == 0:
                    self.count += 1
                    self.flag = 1
            else:
                temp += 1
                if temp == len(contours):
                    self.flag = 0

        return gray_blank

    def run(self):
        while True:
            ret, frame = self.video.read()
            if not ret:
                print("Can't read frame")
                break

            frame = cv.resize(frame, self.frame_size)
            mask = self.preprocess(frame)
            self.draw_lines(frame)

            gray_blank = self.count_objects(mask, frame)

            cv.putText(frame, f'Object = {self.count}', (50, 50),
                       cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

            cv.imshow('Video', gray_blank)
            cv.imshow('original', frame)

            key = cv.waitKey(1)
            if key == ord('q'):
                break

        self.video.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    video_path = '/home/wot-suzal/internship/obj_counting/counting_diff_obj/bottle_test.mp4'
    counter = ObjectCounter(video_path)
    counter.run()
