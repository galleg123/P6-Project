import cv2
import numpy as np

np.seterr(divide='ignore', invalid='ignore')


class Preprocessing:
    def __init__(self, input_video_path, frame_num, threshold=12.5, min_aspect_ratio=0.5):
        self.input_video_path = input_video_path
        self.threshold = threshold
        self.frame_num = frame_num
        self.min_aspect_ratio = min_aspect_ratio
        self.frame = None
        self.mask = None
        self.run()

    def run(self):
        self.last_frame = None
        self.new_frame = None
        self.cam = None
        self.cam = cv2.VideoCapture(self.input_video_path)

        self.cam.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num - 1)
        ret1, self.last_frame = self.cam.read()
        ret2, self.new_frame = self.cam.read()

        # Motion detection to find ROI
        self.last_frame = self._preprocess_frame(self.last_frame)
        self.new_frame2 = self._preprocess_frame(self.new_frame)
        # Find absolute diff
        frame_diff = cv2.absdiff(src1=self.last_frame, src2=self.new_frame2)
        # Threshold the values so that only large enough difference is left
        frame_threshold = cv2.threshold(src=frame_diff, thresh=self.threshold, maxval=255, type=cv2.THRESH_BINARY)[1]
        # Do closing to remove gaps and get a large blob
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        closed_img = cv2.morphologyEx(frame_threshold, cv2.MORPH_CLOSE, kernel)
        # Find the contours of the binary image
        contours, hierarchy = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)==0:
            self.frame = np.zeros((720, 1280), dtype=np.uint8)
            return
        # Only use the largest contour
        contour = sorted(contours, key=cv2.contourArea)[-1]
        #print(cv2.contourArea(contour))
        if cv2.contourArea(contour)<1000:
            self.frame = np.zeros((720, 1280), dtype=np.uint8)
            return
        # The ROI will be the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(contour)

        # Now using the ROI to detect cage blobs
        #print(f"{y}:{y + h}, {x}:{x + w}")
        croppedImg = cv2.cvtColor(self.new_frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
        hue = croppedImg[:, :, 0]

        # Find the most frequent color range
        counts, bins = np.histogram(hue, bins=20)
        # Only take the most frequent color in ROI
        maxCountIndex = np.argmax(counts)
        minRange = bins[maxCountIndex]
        maxRange = bins[maxCountIndex + 1]
        frame_threshold2 = cv2.inRange(hue, minRange, maxRange)
        # Do morphology to close small gaps and then remove irregular shapes and noise
        kernel2 = np.ones((10, 10), np.uint8)
        kernel22 = np.ones((40, 40), np.uint8)
        closed_img2 = cv2.morphologyEx(frame_threshold2, cv2.MORPH_CLOSE, kernel2)
        opened_img2 = cv2.morphologyEx(closed_img2, cv2.MORPH_OPEN, kernel22)
        self.frame = np.zeros((720, 1280), dtype=np.uint8)
        #print(f"{y}:{y+h}, {x}:{x+w}")
        #print(np.shape(self.frame), np.shape(opened_img2))
        self.frame[y:y+h, x:x+w] = opened_img2

        # blob_img, mask = self._blob_detection(edges)
        # self.frame = self._remove_singular_pixels(blob_img)
        # self.mask = mask
        # print(f'Done with frame: {self.frame_num} in video: {self.input_video_path}')
        self._cleanup()

    def _preprocess_frame(self, frame):
        try:
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            processed_frame = cv2.GaussianBlur(src=processed_frame, ksize=(5, 5), sigmaX=0)
        except Exception as e:
            print(f'\t\tError occured in frame: {self.frame_num}, video: {self.input_video_path}: {e}')
            processed_frame = np.zeros((1280, 720, 1), dtype=np.uint8)

        return processed_frame

    def _get_edges(self, frame):
        frame_threshold = cv2.threshold(src=frame, thresh=self.threshold, maxval=255, type=cv2.THRESH_BINARY)[1]
        cv2.imshow('frameThresh', frame_threshold);
        cv2.waitKey(0)
        edges = cv2.Canny(frame_threshold, 1, 128, apertureSize=3)
        cv2.imshow('frameEdges', edges);
        cv2.waitKey(0)
        return edges

    def _remove_singular_pixels(self, img):
        # Create a kernel for dilation
        kernel = np.ones((80, 80), np.uint8)
        kernel1 = np.ones((10, 10), np.uint8)
        # Dilate the image to connect neighboring pixels
        dilated = cv2.dilate(img, kernel)

        # Erode the dilated image to remove single pixels
        eroded = cv2.erode(dilated, kernel1)

        # Apply morphological opening operation
        opened_img = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)

        return opened_img

    def _blob_detection(self, img):
        color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Convert to HSV color space
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

        # Define the range of colors to threshold
        lower_color = np.array([0, 0, 0])  # black color
        upper_color = np.array([200, 255, 200])  # white/gray colors

        # Create a mask based on the color range
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Invert the mask to set the background to black and items to white
        inverted_mask = cv2.bitwise_not(mask)

        # Return the segmented image
        return inverted_mask, mask

    def _cleanup(self):
        self.cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # input_video_path = 'cages/cage1_red_empty.avi'
    # input_video_path = 'dataset/people/people_with_hvis_control.avi'
    input_video_path = '../dataset/cages/cage1_red_empty.avi'
    # input_video_path = 'Machine_Learning/Modules/cage1_red_empty.avi'
    preprocessing = Preprocessing(input_video_path, 122, threshold=20)
    print(preprocessing.frame)
    # Display the frame
    if preprocessing.frame is not None:
        cv2.imshow('Frame', preprocessing.frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
