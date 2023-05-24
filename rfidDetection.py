import cv2
import numpy as np
from classes.motionDetection_v2 import motion_detector
from classes.cageDetection import cage_detector
import cProfile
import pstats

if __name__ == "__main__":
    #input_video = 'cage1_red_empty.avi'
    #input_video = 'people_with_hvis_control.avi'
    camera = cv2.VideoCapture(0)
    camera.set(3, 1280)
    camera.set(4, 720)
    camera.set(cv2.CAP_PROP_FPS, 10)
    first = True
    cageDetector = cage_detector()
    while True:
        pr = cProfile.Profile()
        pr.enable()
        ret, frame = camera.read()
        if first:
            motionDetector = motion_detector(frame,50)
            first=False
            print(f'Frame: {camera.get(cv2.CAP_PROP_POS_FRAMES)}\n\tMotion:\tFalse\n\tCage:\tFalse\n')
            continue

        if not ret:
            break
            
        key = cv2.waitKey(1)
        motion = motionDetector.compare_frames(frame)
        if isinstance(motion, np.ndarray):
          cage, frame, blob  = cageDetector.detect_cage(motion, frame)
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
        pr.disable()
        # Print profiling stats
        ps = pstats.Stats(pr)
        ps.sort_stats(pstats.SortKey.TIME)
        #ps.print_stats(30)
