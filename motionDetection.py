import cv2 as cv
import numpy as np
import time


###### Camera input here #######







################################



class motion_detector():

    def __init__(self, threshold, detected_timeout):

        self.threshold = threshold
        self.time_last = time.perf_counter()
        self.detected = False
        self.detect_timeout = detected_timeout
        self.detect_timer = 0
        self.prev_frame = None
        self.running = True
        input_video = "15FPS_720PL.mp4"

        self.cam = cv.VideoCapture(input_video)

        
    def image_grabber(self):
        return self.cam.read()

    def run(self):
        while self.running:
            

            ret, frame = self.image_grabber()

            self.frame = frame

            if not ret:
                break

            


            processed_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

            processed_frame = cv.GaussianBlur(src=processed_frame, ksize=(5,5), sigmaX=0)

            self.check_timer()
            # If this is the first captured frame, save it and do nothing else.
            if (self.prev_frame is None):
                self.prev_frame = processed_frame
                continue            
                
            # Update frame in case it is still in timeout
            if self.detect_timer <= self.detect_timeout:
                self.prev_frame = processed_frame
                #continue
            
            self.compare_frames(processed_frame)

        
    def check_timer(self):
        self.detect_timer = time.perf_counter() - self.time_last


    def compare_frames(self, new_frame):
            

        # Calculation of the difference between the two frames
        frame_diff = cv.absdiff(src1 = self.prev_frame, src2 = new_frame)
        
        largeEnough = False # Parameter set to True if a large enough area of motion is detected

        # Dilution to make the difference more visible
        kernel = np.ones((5,5))
        frame_diff = cv.dilate(frame_diff, kernel, 1)

        # Threshold to only take big enough differences
        frame_threshold = cv.threshold(src=frame_diff, thresh=self.threshold, maxval=255, type=cv.THRESH_BINARY)[1]


        contours, _ = cv.findContours(image=frame_threshold, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv.contourArea(contour) < 5000:
                # too small: skip!
                continue
            largeEnough = True
            (x, y, w, h) = cv.boundingRect(contour)
            cv.rectangle(img=self.frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

        cv.imshow('Motion detector', self.frame)

        if (cv.waitKey(66) == 27):
            cv.destroyAllWindows()
            self.running = False

        if largeEnough:
            self.time_last = time.perf_counter()

        self.prev_frame = new_frame
        

if __name__ == "__main__":
    motionDetecter = motion_detector(10, 1)

    motionDetecter.run()