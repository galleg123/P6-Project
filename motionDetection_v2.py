import cv2 as cv
import numpy as np
import time


###### Camera input here #######







################################



class motion_detector():

    def __init__(self, threshold):

        self.threshold = threshold
        self.time_last = time.perf_counter()
        self.detected = False
        self.detect_timer = 0
        self.prev_frame = None
        self.running = True
        input_video = "test_video.avi"

        self.cam = cv.VideoCapture(input_video)

        
    def image_grabber(self):
        return self.cam.read()

    def run(self):
        while self.running:
            

            ret, frame = self.image_grabber()

            self.frame = frame

            if not ret:
                break


            # Convert to grayscale
            processed_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)


            # If this is the first captured frame, save it and do nothing else.
            if (self.prev_frame is None):
                self.prev_frame = processed_frame
                continue

                
            self.compare_frames(processed_frame)


    def compare_frames(self, new_frame):
            

        # Calculation of the difference between the two frames
        frame_diff = cv.absdiff(src1 = self.prev_frame, src2 = new_frame)
        

        # Threshold to only take big enough differences
        frame_threshold = cv.threshold(src=frame_diff, thresh=self.threshold, maxval=1, type=cv.THRESH_BINARY)[1]

        # Sum of threshold must be above value to be considered motion
        if np.sum(frame_threshold)/(1920*720) > 0.005: # change this value to change the amount of pixel which have to be changed
            if self.detected==False:
                self.detected=True
                print(f'Detected: {self.detected}')
        else:
            if self.detected == True:
                self.detected = False
                print(f'Detected: {self.detected}')

        #print(f'{np.sum(frame_threshold)/(1920*720)} % of area is filled')

        cv.imshow('Motion detector', self.frame)

        if (cv.waitKey(100) == 27):
            cv.destroyAllWindows()
            self.running = False

        self.prev_frame = new_frame
        

if __name__ == "__main__":
    motionDetecter = motion_detector(20)

    motionDetecter.run()