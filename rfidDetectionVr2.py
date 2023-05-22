import cv2
import numpy as np
from classes.motionDetection_v2 import motion_detector
from classes.cageDetection import cage_detector
import time

if __name__ == "__main__":

    start_time = time.time()

    # Duration of the video in seconds
    video_duration = 3600
    # Duration of the first and last minutes to record in seconds
    record_duration = 60

    camera = cv2.VideoCapture(1)

    camera.set(3, 1280)
    camera.set(4, 720)

    # Get the frames per second (fps) and total frame count of the video
    camera.set(cv2.CAP_PROP_FPS, 10)
    fps = camera.get(cv2.CAP_PROP_FPS)
    frame_count = int(fps * video_duration)

    # Calculate the frame index ranges for the first and last minutes
    first_minute_start = 0
    first_minute_end = int(fps * record_duration)
    last_minute_start = frame_count - int(fps * record_duration)
    last_minute_end = frame_count
    frame_index = 0

    # Create a VideoWriter object to save the recorded segments
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_first = cv2.VideoWriter('output_first_video.avi', fourcc, fps, (1280, 720))
    output_last = cv2.VideoWriter('output_last_video.avi', fourcc, fps, (1280, 720))

    first = True
    cageDetector = cage_detector()
    while True:
        ret, frame = camera.read()
        if first:
            motionDetector = motion_detector(frame,50)
            first=False
            print(f'Frame: {camera.get(cv2.CAP_PROP_POS_FRAMES)}\n\tMotion:\tFalse\n\tCage:\tFalse\n')
            continue

        if not ret:
            break

        print("--- %s seconds ---" % (time.time() - start_time))

        if first_minute_start <= frame_index < first_minute_end:
            # Record frames from the first minute
            output_first.write(frame)
        if last_minute_start <= frame_index < last_minute_end:
            # Record frames from the last minute
            output_last.write(frame)

        motion = motionDetector.compare_frames(frame)
        if isinstance(motion, np.ndarray):
          cage, frame = cageDetector.detect_cage(motion, frame)
          print(f'Frame: {camera.get(cv2.CAP_PROP_POS_FRAMES)}\n\tMotion:\tTrue\n\tCage:\t{cage}\n')
        else:
          print(f'Frame: {camera.get(cv2.CAP_PROP_POS_FRAMES)}\n\tMotion:\tFalse\n\tCage:\tFalse\n')

        cv2.imshow("Video", frame)
        key = cv2.waitKey(100)
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

        if frame_index >= int(fps * video_duration):
            break

        frame_index += 1

    # Release the video capture and writer objects
    camera.release()
    output_first.release()
    output_last.release()

    # Destroy any OpenCV windows
    cv2.destroyAllWindows()