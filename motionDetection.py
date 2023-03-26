from logging import _srcfile
import cv2 as cv


###### Camera input here #######
"""
def image_grabber():
    return camera_input


"""

################################



def motion_detector():

    threshold = 20

    frame_count = 0
    prev_frame = None

    while True:
        frame_count += 1

        frame = image_grabber()

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
            if cv.contourArea(contour) < 50:
                # too small: skip!
                continue
            (x, y, w, h) = cv.boundingRect(contour)
            cv.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

        cv.imshow('Motion detector', frame)

        if (cv.waitKey(30) == 27):
            break