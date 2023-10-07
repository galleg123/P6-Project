import cv2
import numpy as np
from classes.motionDetection_v2 import motion_detector
from classes.cageDetection import cage_detector
import time
import csv

if __name__ == "__main__":
    input_video = 'cage1_red_empty.avi'
    #input_video = 'people_with_hvis_control.avi'
    #camera = cv2.VideoCapture(0)
    camera = cv2.VideoCapture(input_video)
    camera.set(3, 1280)
    camera.set(4, 720)
    camera.set(cv2.CAP_PROP_FPS, 10)
    first = True
    cageDetector = cage_detector()
    csv_file_path = "time_taking_with_convex.csv"
    frame_counter = 0
    while True:
        # Measure main code runtime
        main_start_time = time.time()

        ret, frame = camera.read()
        if first:
            motionDetector = motion_detector(frame,50)
            first=False
            print(f'Frame: {camera.get(cv2.CAP_PROP_POS_FRAMES)}\n\tMotion:\tFalse\n\tCage:\tFalse\n')
            frame_counter +=1
            with open(csv_file_path, "w") as file:
                file.write("motion_detector,main,cage_detector\n")
            continue

        if not ret:
            break
            
        key = cv2.waitKey(1)
        motion_start_time = time.time()
        motion = motionDetector.compare_frames(frame)
        motion_end_time = time.time()
        detector_start_time = None
        if isinstance(motion, np.ndarray):
          detector_start_time = time.time()
          cage, frame, blob  = cageDetector.detect_cage(motion, frame)
          detector_end_time = time.time()
          print(f'Frame: {camera.get(cv2.CAP_PROP_POS_FRAMES)}\n\tMotion:\tTrue\n\tCage:\t{cage}\n')
          cv2.imshow("blob", blob)
        else:
          print(f'Frame: {camera.get(cv2.CAP_PROP_POS_FRAMES)}\n\tMotion:\tFalse\n\tCage:\tFalse\n')
        cv2.imshow(f'coolio     ',frame)
        #cv2.imshow(f'{input_video}', frame)
        
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
                    cur_frame_number = camera.get(cv2.CAP_PROP_POS_FRAMES)
                    print('* At frame #' + str(cur_frame_number))

                    prev_frame = cur_frame_number
                    if (cur_frame_number > 1):
                        prev_frame -= 10

                    print('* Rewind to frame #' + str(prev_frame))
                    camera.set(cv2.CAP_PROP_POS_FRAMES, prev_frame)

                if key == ord('q'):
                    break
        main_end_time = time.time()
        main_runtime = main_end_time - main_start_time
        motion_runtime = motion_end_time - motion_start_time
        if detector_start_time is None: detector_runtime = ""
        else: detector_runtime = detector_end_time - detector_start_time
        """
        with open(csv_file_path, 'r') as file:
            reader = csv.reader(file)
            data = list(reader)

        main_column = -1
        if data and "main" in data[0]:
            main_column = data[0].index("main")
        
        if main_column == -1:
            # Add "main" column
            data[0].append("main")
            main_column = len(data[0]) - 1
        
        if len(data) == 1:
            # If there is only one row, add a new row with the runtime values
            data.append([None] * len(data[0]))
            data[-1][main_column] = main_runtime
        else:
            # If there are existing rows, update the last row with the runtime values
            last_row_index = len(data) - 1
            if len(data[last_row_index]) <= main_column:
                # If the "main" column doesn't exist in the last row, append it
                data[last_row_index].append(main_runtime)
            else:
                # Otherwise, modify the existing value
                data[last_row_index][main_column] = main_runtime

        # Write the modified data back to the CSV file
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        """
        with open(csv_file_path, "a") as file:
            file.write(f"{motion_runtime},{main_runtime},{detector_runtime}\n")

        frame_counter += 1
        if frame_counter == 1000:
            break
