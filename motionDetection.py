import cv2 as cv
import numpy as np
import time


###### Camera input here #######

input_video = "15FPS_720PL.mp4"

cam = cv.VideoCapture(0)

def image_grabber():
    return cam.read()



################################



def motion_detector():

    threshold = 10
    time_now = time.perf_counter()
    time_last = time.perf_counter()
    detected = False

    frame_count = 0
    prev_frame = None

    while True:

        frame_count += 1

        largeEnough = False # Parameter set to True if a large enough area of motion is detected

        ret, frame = image_grabber()

        if not ret:
            break

        processed_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        processed_frame = cv.GaussianBlur(src=processed_frame, ksize=(5,5), sigmaX=0)

        # If this is the first captured frame, save it and do nothing else.
        if (prev_frame is None):
            prev_frame = processed_frame
            continue


        # Calculation of the difference between the two frames
        frame_diff = cv.absdiff(src1 = prev_frame, src2 = processed_frame)
        prev_frame = processed_frame


        # Dilution to make the difference more visible
        kernel = np.ones((5,5))
        frame_diff = cv.dilate(frame_diff, kernel, 1)

        # Threshold to only take big enough differences
        frame_threshold = cv.threshold(src=frame_diff, thresh=threshold, maxval=255, type=cv.THRESH_BINARY)[1]


        contours, _ = cv.findContours(image=frame_threshold, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv.contourArea(contour) < 5000:
                # too small: skip!
                continue
            largeEnough = True
            (x, y, w, h) = cv.boundingRect(contour)
            cv.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

        cv.imshow('Motion detector', frame)

        if (cv.waitKey(66) == 27):
            break

        if largeEnough:
            if not detected:
                print(f'Movement detected, there was no movement for {round(time.perf_counter()-time_last,3)} seconds.')
                time_last = time.perf_counter()
            detected = True
        else:
            if detected:
                print(f'Movement stopped, movement lasted for {round(time.perf_counter()-time_last,3)} seconds.')
                time_last = time.perf_counter()
            detected = False

if __name__ == "__main__":
    motion_detector()