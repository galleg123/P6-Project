import cv2
import numpy as np
import time
import csv

class motion_detector:

    def __init__(self, first_frame, threshold):
        self.threshold = threshold
        self.prev_frame = first_frame
        self.prev_frame = cv2.cvtColor(self.prev_frame, cv2.COLOR_RGB2GRAY)

    def compare_frames(self, new_frame):
        csv_file_path = "time_taking.csv"
        motion_start_time = time.time()
        # Convert to grayscale
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
        
        # Calculation of the difference between the two frames
        frame_diff = cv2.absdiff(src1 = self.prev_frame, src2 = new_frame)

        # Threshold to only take big enough differences
        frame_threshold = cv2.threshold(src=frame_diff, thresh=self.threshold, maxval=255, type=cv2.THRESH_BINARY)[1]
        self.prev_frame = new_frame
        motion_end_time = time.time()
        motion_detector_runtime = motion_end_time - motion_start_time
        file_exists = False
        try:
            with open(csv_file_path, 'r') as file:
                file_exists = True
        except FileNotFoundError:
            pass

        if not file_exists:
            # If the file doesn't exist, create a new CSV file with the "main" runtime value
            data = [["motion_detector"]]
            data.append([motion_detector_runtime])
        else:
            # Read the existing CSV data
            data = []
            with open(csv_file_path, 'r') as file:
                reader = csv.reader(file)
                data = list(reader)
             # Check if "main" column already exists
            motion_detector_column = -1
            if data and "motion_detector" in data[0]:
                motion_detector_column = data[0].index("motion_detector")

            if motion_detector_column == -1:
                # Add "motion_detector" column
                data[0].append("motion_detector")
                motion_detector_column = len(data[0]) - 1

            # Add a new row with the "motion_detector" runtime value
            data.append([None] * len(data[0]))
            data[-1][motion_detector_column] = motion_detector_runtime

        # Write the modified data back to the CSV file
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        # Sum of threshold must be above value to be considered motion
        if np.sum(frame_threshold)/(1280*720) > (0.75): # change this value to change the amount of pixel which have to be changed
            return frame_threshold
        else:
            return False
        
