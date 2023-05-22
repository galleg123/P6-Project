import cv2
import numpy as np
import cProfile
import pstats


np.seterr(divide='ignore', invalid='ignore')


class cage_detector:
    def __init__(self, testing=True, performance=False):
        self.testing = testing 
        self.performance = performance
        self.new_width = 640
        self.new_height = 480
        self.cage = False
        

    def detect_cage(self, motion, frame):
        pr = cProfile.Profile()
        pr.enable()
        if self.performance:
            resized_frame = self.resize_frame(motion)
        edges = self.get_edges(motion)
        blob_img, mask = self.blob_detection(edges)
        blob_img_classified, params_dicts = self.blob_classifier(blob_img, mask)
        
        # Uncomment this section to see if a cage is detected
        if params_dicts and self.testing:
            blob_img_classified = self.detected_cage(blob_img_classified, params_dicts)
            frame = self.detected_cage(frame, params_dicts)
        pr.disable()
        # Print profiling stats
        ps = pstats.Stats(pr)
        ps.sort_stats(pstats.SortKey.TIME)
        ps.print_stats(10)
        
        return self.cage, blob_img_classified

    def resize_frame(self, frame):
        return cv2.resize(frame, (self.new_width, self.new_height))

    def preprocess_frame(self, frame):
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        processed_frame = cv2.GaussianBlur(src=processed_frame, ksize=(5, 5), sigmaX=0)

        if self.prev_frame is None:
            self.prev_frame = processed_frame
            return processed_frame

        return processed_frame

    def get_edges(self, frame):
        frame_threshold = cv2.threshold(src=frame, thresh=12.5, maxval=255, type=cv2.THRESH_BINARY)[1]
        edges = cv2.Canny(frame_threshold, 1, 128, apertureSize=3)
        return edges

    def remove_singular_pixels(self, img):
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

    def blob_detection(self, img):
        color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        color = self.remove_singular_pixels(color)
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

    def threshold_calc(self, contour, blob_img):
        """
        Calculate the threshold value of the contour
        """
        x, y, w, h = cv2.boundingRect(contour)
        roi = blob_img[y:y + h, x:x + w]
        threshold_value = cv2.mean(roi)[0]
        return threshold_value

    def circularity_calc(self, contour, area):
        """Calculate the circularity of the contour"""
        try:
            perimeter = cv2.arcLength(contour, True)
            return (2 * np.sqrt(np.pi * area)) / perimeter
        except cv2.error:
            print("Error: Failed to compute circularity for contour")
            return None

    def convexity_calc(self, contour):
        """Calculate the convexity of the contour"""
        try:
            perimeter = cv2.arcLength(contour, True)
            hull = cv2.convexHull(contour)
            hull_perimeter = cv2.arcLength(hull, True)
            return hull_perimeter / perimeter
        except cv2.error:
            print("Error: Failed to compute convexity for contour")
            return None

    def elongation_calc(self, contour):
        """Calculate the elongation of a contour"""
        try:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= h:
                return h / w
            else:
                return w / h
        except cv2.error:
            print("Error: Failed to compute elongation for contour")
            return None

    def eccentricity_calc(self, contour):
        """Calculate the eccentricity of a contour"""
        try:
            (x, y), (a, b), angle = cv2.fitEllipse(contour)
            if a >= b:
                return np.sqrt(1 - (b / a) ** 2)
            else:
                return np.sqrt(1 - (a / b) ** 2)
        except cv2.error:
            print("Error: Failed to compute eccentricity for contour")
            return None

    def solidity_calc(self, contour, area):
        """Calculate the solidity of the contour"""
        try:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            return area / hull_area
        except cv2.error:
            print("Error: Failed to compute solidity for contour")
            return None

    def rectangularity_rotate_calc(self, contour, area):
        """Calculate the rectangularity of the contour"""
        try:
            rect = cv2.minAreaRect(contour)
            rect_area = rect[1][0] * rect[1][1]
            rectangularity = area / rect_area
            return rectangularity
        except cv2.error:
            print("Error: Failed to compute rectangularity for contour")
            return None

    def blob_classifier(self, blob_img, mask, font=cv2.FONT_HERSHEY_SIMPLEX):

        # Define the parameters to calculate for each blob
        params = ["Area", "Circularity", "Eccentricity", "Elongation", "Convexity", "Rectangularity", "Solidity",
                  "Cage"]

        # Find the contours of the binary image
        contours, hierarchy = cv2.findContours(blob_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the blob image
        blob_img_copy = blob_img.copy()

        # Define an empty list to store the positions of the text
        text_positions = []

        params_dicts = []
        # Loop through each contour and calculate the parameters
        for i, contour in enumerate(contours):
            # Calculate the area of the contour
            area = cv2.contourArea(contour)

            # Calculate the circularity of the contour
            circularity = self.circularity_calc(contour, area)

            # Calculate the Eccentricity of the contour
            eccentricity = self.eccentricity_calc(contour)

            # Calculate the elongation of the contour
            elongation = self.elongation_calc(contour)

            # Calculate the convexity of the contour
            convexity = self.convexity_calc(contour)

            # Calculate the rectangularity of the contour
            rectangularity = self.rectangularity_rotate_calc(contour, area)

            # Calculate the solidity of the contour
            solidity = self.solidity_calc(contour, area)

            # Checking if it should detect cage and create a dictionary of the parameters
            if self.testing:
                # Insert trained model here: 
                if (area > 120000 and rectangularity > 0.7) and (area > 120000 and convexity > 0.9):
                    self.cage = True

                params_dict = {"Area": area, "Circularity": circularity,
                               "Eccentricity": eccentricity, "Elongation": elongation,
                               "Convexity": convexity, "Rectangularity": rectangularity, 
                               "Solidity": solidity, "Cage": self.cage}
            else:
                params_dict = {"Area": area, "Circularity": circularity,
                               "Eccentricity": eccentricity, "Elongation": elongation,
                               "Convexity": convexity, "Rectangularity": rectangularity, 
                               "Solidity": solidity}
            
            # Draw the contour on the blob image
            cv2.drawContours(blob_img_copy, [contour], -1, (255, 255, 255), thickness=-1)
            # Convert the grayscale image to RGB
            if blob_img_copy.ndim == 2:
                blob_img_copy = cv2.cvtColor(blob_img_copy, cv2.COLOR_GRAY2RGB)

            # Display the parameters under the contour
            x, y, w, h = cv2.boundingRect(contour)
            h = max(h, 50)  # increase the height to 50 pixels
            cx, cy = x + w // 2, y + h // 2
            collision = False
            for j in range(i):
                if text_positions and j < len(text_positions):
                    x2, y2, w2, h2 = text_positions[j]
                    if cx < x2 + w2 and cx + w // 2 > x2 and cy < y2 + h2 and cy + h // 2 > y2:
                        collision = True
                        break

            # If there is no collision, add the text position to the list
            if not collision:
                text_positions.append((cx - w // 2, cy - h // 2, w, h))

                # Draw the text on the blob image
                cv2.putText(blob_img_copy, f"{i + 1}", (cx, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 1)
                for j, param in enumerate(params):
                    if param in params_dict:
                        text = params_dict[param]
                        cv2.putText(blob_img_copy, f'{param}: {text}', (cx, cy + 15 + j * 15), font, 0.4, (0, 128, 0),
                                    1)
            else:

                # Draw the text on the blob image
                cv2.putText(blob_img_copy, f"{i + 1}", (cx, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 1)
                for j, param in enumerate(params):
                    if param in params_dict:
                        text = params_dict[param]
                        cv2.putText(blob_img_copy, f'{param}: {text}', (cx, cy + 15 + j * 15), font, 0.4, (0, 128, 0),
                                    1)
                        collision = False

            params_dicts.append(params_dict)

        return blob_img_copy, params_dicts

    def detected_cage(self, frame, params_dicts):
        cv2.rectangle(frame, (int(frame.shape[1] / 2 - 100), 0), (int(frame.shape[1] / 2), 100), (0, 0, 255), -1)
        text = "Movement detected"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = int((frame.shape[1] / 3 - text_size[0]) / 2)
        text_y = int(frame.shape[0] - text_size[1] - 10)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

        if self.cage:
            cv2.rectangle(frame, (int(frame.shape[1] / 2), 0), (int(frame.shape[1] / 2 + 100), 100), (0, 255, 0), -1)
            text = "Rolling cage detected"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_x = int((frame.shape[1] + 800 - text_size[0]) / 2)
            text_y = int(frame.shape[0] - text_size[1] - 10)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
        return frame

    def cleanup(self):
        self.cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    input_video_path = 'dataset/cages/cage1_red_empty.avi'
    #input_video_path = 'dataset/people/people_with_hvis_control.avi'
