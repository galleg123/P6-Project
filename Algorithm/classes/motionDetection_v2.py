import cv2
import numpy as np


class motion_detector:

    def __init__(self, first_frame, threshold):
        self.threshold = threshold
        self.prev_frame = first_frame
        self.prev_frame = cv2.cvtColor(self.prev_frame, cv2.COLOR_RGB2GRAY)

    def compare_frames(self, new_frame):

        # Convert to grayscale
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
        
        # Calculation of the difference between the two frames
        frame_diff = cv2.absdiff(src1 = self.prev_frame, src2 = new_frame)

        # Threshold to only take big enough differences
        frame_threshold = cv2.threshold(src=frame_diff, thresh=self.threshold, maxval=255, type=cv2.THRESH_BINARY)[1]
        self.prev_frame = new_frame

        # Sum of threshold must be above value to be considered motion
        if np.sum(frame_threshold)/(1280*720) > (0.75): # change this value to change the amount of pixel which have to be changed
            return frame_threshold
        else:
            return False
        
