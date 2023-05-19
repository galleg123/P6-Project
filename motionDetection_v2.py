import cv2
import numpy as np


class motion_detector():

    def __init__(self, threshold):

        self.threshold = threshold
        

    def compare_frames(self, prev_frame, new_frame):
            
        # Convert to grayscale
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)

        # Calculation of the difference between the two frames
        frame_diff = cv2.absdiff(src1 = prev_frame, src2 = new_frame)
        

        # Threshold to only take big enough differences
        frame_threshold = cv2.threshold(src=frame_diff, thresh=self.threshold, maxval=1, type=cv2.THRESH_BINARY)[1]

        # Sum of threshold must be above value to be considered motion
        if np.sum(frame_threshold)/(1920*720) > 0.005: # change this value to change the amount of pixel which have to be changed
            return frame_diff
        else:
            return False

        
