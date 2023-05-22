import cv2
import numpy as np
from classes.motionDetection_v2 import motion_detector
from classes.cageDetection import cage_detector

if __name__ == "__main__":
  input_video = 'cage1_red_empty.avi'
  camera = cv2.VideoCapture(input_video)
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

    motion = motionDetector.compare_frames(frame)
    if isinstance(motion, np.ndarray):
      cage, frame = cageDetector.detect_cage(motion, frame)
      print(f'Frame: {camera.get(cv2.CAP_PROP_POS_FRAMES)}\n\tMotion:\tTrue\n\tCage:\t{cage}\n')
    else:
      print(f'Frame: {camera.get(cv2.CAP_PROP_POS_FRAMES)}\n\tMotion:\tFalse\n\tCage:\tFalse\n')

    cv2.imshow(f'{input_video}', frame)
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
