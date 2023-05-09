import cv2 as cv2
import numpy as np
import cProfile
import pstats


np.seterr(divide='ignore', invalid='ignore')

class CageDetector:
    def __init__(self, input_video_path, threshold=12.5, min_aspect_ratio=0.5):
        self.input_video_path = input_video_path
        self.threshold = threshold
        self.min_aspect_ratio = min_aspect_ratio
        self.frame_count = 0
        self.new_width = 640
        self.new_height = 480


    def run(self):
        self.prev_frame = None
        self.cam = None
        self.cam = cv2.VideoCapture(self.input_video_path)
        # Start profiling
        pr = cProfile.Profile()
        pr.enable()

        while True:
            self.frame_count += 1
            ret, frame = self.cam.read()

            if not ret:
                break
            
            resized_frame = self._resize_frame(frame)
            processed_frame = self._preprocess_frame(frame)
            frame_diff = cv2.absdiff(src1=self.prev_frame, src2=processed_frame)
            self.prev_frame = processed_frame

            edges = self._get_edges(frame_diff)
            blob_img, mask = self._blob_detection(edges)
            blob_img_cleaned = self._remove_singular_pixels(blob_img)
            blob_img_classified, params_dicts = self._blob_classifier(blob_img_cleaned, mask)            
            
            if params_dicts:
                blob_img_classified = self._detected_cage(blob_img_classified, params_dicts)
                frame = self._detected_cage(frame, params_dicts)
            #cv2.imshow('Frame Difference', frame_diff)
            #cv2.imshow('Mask', mask)
            #cv2.imshow('Edge Definition', edges)
            cv2.imshow('Blob Classifier', blob_img_classified)
            #cv2.imshow('New Frame Processed', processed_frame)
            #cv2.imshow('Last Frame Processed', self.prev_frame)
            #cv2.imshow('Original', frame)
            key = cv2.waitKey(10)
            if key == ord('q'):
                break
             # check if 'p' was pressed and wait for a 'b' press
    
            if (key & 0xFF == ord('p')):

                # sleep here until a valid key is pressed
                while (True):
                    key = cv2.waitKey(0)

                    # check if 'p' is pressed and resume playing
                    if (key & 0xFF == ord('p')):
                        break

                    # check if 'b' is pressed and rewind video to the previous frame, but do not play
                    if (key & 0xFF == ord('b')):
                        cur_frame_number = self.cam.get(cv2.CAP_PROP_POS_FRAMES)
                        print('* At frame #' + str(cur_frame_number))

                        prev_frame = cur_frame_number
                        if (cur_frame_number > 1):
                            prev_frame -= 10

                        print('* Rewind to frame #' + str(prev_frame))
                        self.cam.set(cv2.CAP_PROP_POS_FRAMES, prev_frame)

                    if key == ord('q'):
                        break
        # Stop profiling
        pr.disable()

        # Print profiling stats
        ps = pstats.Stats(pr)
        ps.sort_stats(pstats.SortKey.TIME)
        ps.print_stats(10)

        self._cleanup()

    def _resize_frame(self, frame):
        return cv2.resize(frame, (self.new_width, self.new_height))

    def _preprocess_frame(self, frame):
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        processed_frame = cv2.GaussianBlur(src=processed_frame, ksize=(5, 5), sigmaX=0)

        if self.prev_frame is None:
            self.prev_frame = processed_frame
            return processed_frame

        return processed_frame

    def _get_edges(self, frame):
        frame_threshold = cv2.threshold(src=frame, thresh=self.threshold, maxval=255, type=cv2.THRESH_BINARY)[1]
        edges = cv2.Canny(frame_threshold, 1,128, apertureSize=3)
        return edges

    def _remove_singular_pixels(self, img):
        # Create a kernel for dilation
        kernel = np.ones((100, 100), np.uint8)

        # Dilate the image to connect neighboring pixels
        dilated = cv2.dilate(img, kernel)

        # Erode the dilated image to remove single pixels
        eroded = cv2.erode(dilated, kernel)

        # Apply morphological opening operation
        opened_img = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)


        return opened_img

    def _blob_detection(self, img):
        color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Convert to HSV color space
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

        # Define the range of colors to threshold
        lower_color = np.array([0, 0, 0]) # black color
        upper_color = np.array([200, 255, 200]) # white/gray colors

        # Create a mask based on the color range
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Invert the mask to set the background to black and items to white
        inverted_mask = cv2.bitwise_not(mask)

        # Return the segmented image
        return inverted_mask, mask

    def _threshold_calc(self, contour, blob_img):
        """
        Calculate the threshold value of the contour
        """
        x, y, w, h = cv2.boundingRect(contour)
        roi = blob_img[y:y+h, x:x+w]
        threshold_value = cv2.mean(roi)[0]
        return threshold_value

    def _circularity_calc(self,contour, area):
        """
        Calculate the circularity of the contour
        """
        perimeter = cv2.arcLength(contour, True)
        if perimeter != 0:
            return 4 * np.pi * area / perimeter ** 2
        else:
            return None

    def _eccentricity_calc(self, contour):
        """Calculate the eccentricity of a contour"""
        if len(contour) < 5:
            print("Error: Contour has less than 5 points")
            return None
        
        try:
            (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
            eccentricity = (1 - (ma / MA) ** 2) ** 0.5
            return eccentricity
        except cv2.error:
            print("Error: Failed to compute eccentricity for contour")
            return None

    
    def _caliper_calc(self,contour):
        """Calculate the caliper diameter of a contour"""
        leftmost = tuple(contour[contour[:,:,0].argmin()][0])
        rightmost = tuple(contour[contour[:,:,0].argmax()][0])
        topmost = tuple(contour[contour[:,:,1].argmin()][0])
        bottommost = tuple(contour[contour[:,:,1].argmax()][0])
        
        d1 = np.linalg.norm(np.array(leftmost) - np.array(rightmost))
        d2 = np.linalg.norm(np.array(topmost) - np.array(bottommost))
        
        caliper_diameter = max(d1, d2)
        
        return caliper_diameter
        
    def _elongation_calc(self,contour):
        """Calculate the elongation of a contour"""
        try:
            (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
            elongation = MA/ma
            return elongation
        except cv2.error:
            #print("Error: Failed to compute elongation for contour")
            return None
    def _inertia_calc(self,contour):
        """
        Calculate the inertia of the contour
        """
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            x_c = moments["m10"] / moments["m00"]
            y_c = moments["m01"] / moments["m00"]
            u20 = moments["mu20"] / moments["m00"] - x_c ** 2
            u02 = moments["mu02"] / moments["m00"] - y_c ** 2
            u11 = moments["mu11"] / moments["m00"] - x_c * y_c
            return (u20 + u02 + np.sqrt(4 * u11 ** 2 + (u20 - u02) ** 2)) / (u20 + u02 - np.sqrt(4 * u11 ** 2 + (u20 - u02) ** 2))
        else:
            return 0

    def _convexity_calc(self,area,mask):
        """
        Calculate the convexity of the contour
        """
        if area > 0:
            indices = np.transpose(np.nonzero(mask))
            if indices.size > 0:
                try:
                    hull = cv2.convexHull(indices)
                    if len(hull) > 0:
                        hull_area = cv2.contourArea(hull)
                        return area / hull_area
                except cv2.error:
                    pass
        return None

    def _blob_classifier(self, blob_img, mask, font=cv2.FONT_HERSHEY_SIMPLEX):
        
        # Define the parameters to calculate for each blob
        params = ["Area", "Threshold", "Circularity", "Eccentricity", "Caliper", "Elongation", "Inertia", "Convexity", "Rectangle"]
        
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
            
            # Calculate the threshold value of the contour
            threshold = self._threshold_calc(contour, blob_img)
            
            # Calculate the circularity of the contour
            circularity = self._circularity_calc(contour, area)

            # Calculate the Eccentricity of the contour
            eccentricity = self._eccentricity_calc(contour)

            # Calculate the caliper of the contour 
            caliper = self._caliper_calc(contour)

            # Calculate the elongation of the contour
            elongation = self._elongation_calc(contour)

            # Calculate the inertia of the contour
            inertia = self._inertia_calc(contour)
            
            # Calculate the convexity of the contour
            #convexity = self._convexity_calc(area,mask)

            # Define the rectangle object
            if (area > 120000 and circularity < 0.8 and circularity > 0.6) or (area < 15000 and area > 10000 and circularity > 0.75):
                rectangle = True
            else:
                rectangle = False
            
            # Create a dictionary of the parameters
            params_dict = {"Area": area, "Threshold": threshold, "Circularity": circularity,"Eccentricity": eccentricity, "Caliper": caliper, "Elongation": elongation, "Inertia": inertia, "Rectangle": rectangle}
            #params_dict = {"Area": area, "Threshold": threshold, "Circularity": circularity, "Inertia": inertia, "Convexity": convexity, "Rectangle": rectangle}
            
            # Draw the contour on the blob image
            cv2.drawContours(blob_img_copy, [contour], -1, (255, 255, 255), thickness=-1)
            # Convert the grayscale image to RGB
            if blob_img_copy.ndim == 2:
                blob_img_copy = cv2.cvtColor(blob_img_copy, cv2.COLOR_GRAY2RGB)
            
            # Display the parameters under the contour
            x, y, w, h = cv2.boundingRect(contour)
            h = max(h, 50)  # increase the height to 50 pixels
            cx, cy = x + w//2, y + h//2
            #cv2.putText(blob_img_copy, f"{i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 1)
            collision = False
            for j in range(i):
                if text_positions and j < len(text_positions):
                    x2, y2, w2, h2 = text_positions[j]
                    if cx < x2 + w2 and cx + w//2 > x2 and cy < y2 + h2 and cy + h//2 > y2:
                        collision = True
                        break

            # If there is no collision, add the text position to the list
            if not collision:
                text_positions.append((cx-w//2, cy-h//2, w, h))
                
                # Draw the text on the blob image
                cv2.putText(blob_img_copy, f"{i+1}", (cx, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 1)
                for j, param in enumerate(params):
                    if param in params_dict:
                        text = params_dict[param]
                        cv2.putText(blob_img_copy, f'{param}: {text}', (cx, cy+15+j*15), font, 0.4, (0, 128, 0), 1)
            else:
                
                # Draw the text on the blob image
                cv2.putText(blob_img_copy, f"{i+1}", (cx, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 1)
                for j, param in enumerate(params):
                    if param in params_dict:
                        text = params_dict[param]
                        cv2.putText(blob_img_copy, f'{param}: {text}', (cx, cy+15+j*15), font, 0.4, (0, 128, 0), 1)
                        collision = False
            
            params_dicts.append(params_dict)
        
        return blob_img_copy, params_dicts

    def _detected_cage(self, frame, params_dicts):
        cv2.rectangle(frame, (int(frame.shape[1]/2-100), 0), (int(frame.shape[1]/2), 100), (0, 0, 255), -1)
        text = "Movement detected"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = int((frame.shape[1]/3 - text_size[0]) / 2)
        text_y = int(frame.shape[0] - text_size[1] - 10)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
        
        if any(sub['Rectangle'] for sub in params_dicts):
            cv2.rectangle(frame, (int(frame.shape[1]/2), 0), (int(frame.shape[1]/2+100), 100), (0, 255, 0), -1)
            
            text = "Rolling cage detected"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_x = int((frame.shape[1]+800 - text_size[0]) / 2)
            text_y = int(frame.shape[0] - text_size[1] - 10)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
        return frame

    def _cleanup(self):
        self.cam.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    input_video_path = 'output.avi'
    CageDetector(input_video_path).run()
