import cv2 as cv
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

class CageDetector:
    def __init__(self, input_video_path, threshold=20, min_aspect_ratio=0.5):
        self.input_video_path = input_video_path
        self.threshold = threshold
        self.min_aspect_ratio = min_aspect_ratio
        self.frame_count = 0
        self.prev_frame = None
        self.cam = None
        self.new_width = 640
        self.new_height = 480


    def run(self):
        self.cam = cv.VideoCapture(self.input_video_path)
        while True:
            self.frame_count += 1
            ret, frame = self.cam.read()

            if not ret:
                break
            
            #resized_frame = self._resize_frame(frame)
            frame = self._square_detection(frame)

            cv.imshow('Motion detector', frame)

            if cv.waitKey(5) & 0xFF == ord('q'):
                break

        self._cleanup()

    def _resize_frame(self, frame):
        return cv.resize(frame, (self.new_width, self.new_height))

    def _preprocess_frame(self, frame):
        processed_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        processed_frame = cv.GaussianBlur(src=processed_frame, ksize=(5, 5), sigmaX=0)

        if self.prev_frame is None:
            self.prev_frame = processed_frame
            return processed_frame

        return processed_frame

    def _get_edges(self, frame):
        frame_threshold = cv.threshold(src=frame, thresh=self.threshold, maxval=255, type=cv.THRESH_BINARY)[1]
        edges = cv.Canny(frame_threshold, 30, 100, apertureSize=3)
        return edges

    def _get_hough_lines(self, edges):
        lines = cv.HoughLines(edges, 1, np.pi/180, 100, min_theta=np.pi/4, max_theta=3*np.pi/4)
        return lines
        """
        if lines is not None:
            lines = lines[:, 0, :]  # convert from 3D to 2D array
            return lines[:, [0, 1]]  # return only rho and theta columns
        else:
            return np.empty((0, 2))  # return an empty 2D array if no lines detected
        """
    def _draw_rectangles(self, lines, frame):
        for line in lines:
            
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2, cv.LINE_AA, shift=0)
            try:
                aspect_ratio = abs(x2-x1) / abs(y2-y1)
            except:
                continue

            if aspect_ratio >= self.min_aspect_ratio and aspect_ratio <= 1/self.min_aspect_ratio:
                cv.rectangle(frame, (frame.shape[1]-100, 0), (frame.shape[1], 100), (0, 255, 0), -1)
                text = "Rolling cage detected"
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
                text_x = int((frame.shape[1] - text_size[0]) / 2)
                text_y = int(frame.shape[0] - text_size[1] - 10)
                cv.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, cv.LINE_AA)
                break
            """
            if len(lines.shape) != 2 or lines.shape[1] != 2:
                return
            rhos, thetas = lines[:, 0], lines[:, 1]
            cos_thetas, sin_thetas = np.cos(thetas), np.sin(thetas)
            xs = np.array([0, frame.shape[1]])
            ys = np.stack((xs[:, np.newaxis] * -sin_thetas, xs[:, np.newaxis] * cos_thetas), axis=-1)
            ys += np.repeat(rhos[:, np.newaxis], 2, axis=1).astype(np.int32)
            ys = np.round(ys).astype(np.int32)
            for y in ys:
                if len(y) != 2:
                    continue
                x1, y1 = y[0]
                x2, y2 = y[1]
                #cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2, cv.LINE_AA, shift=0)
                
            try:
                aspect_ratios = np.abs(ys[:, 1, 1] - ys[:, 0, 1]) / np.abs(ys[:, 1, 0] - ys[:, 0, 0])
            except:
                continue 
                
            valid_lines = (aspect_ratios >= self.min_aspect_ratio) & (aspect_ratios <= 1/self.min_aspect_ratio)
            if any(valid_lines):
                cv.rectangle(frame, (frame.shape[1]-100, 0), (frame.shape[1], 100), (0, 255, 0), -1)
                text = "Rolling cage detected"
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
                text_x = int((frame.shape[1] - text_size[0]) / 2)
                text_y = int(frame.shape[0] - text_size[1] - 10)
                cv.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, cv.LINE_AA)
            """
    def _square_detection(self, frame):
        processed_frame = self._preprocess_frame(frame)
        frame_diff = cv.absdiff(src1=self.prev_frame, src2=processed_frame)
        self.prev_frame = processed_frame

        edges = self._get_edges(frame_diff)
        lines = self._get_hough_lines(edges)

        if lines is not None:
            self._draw_rectangles(lines, frame)
        return frame

    def _cleanup(self):
        self.cam.release()
        cv.destroyAllWindows()

if __name__=="__main__":
    input_video_path = '15FPS_720PL.mp4'
    CageDetector(input_video_path).run()