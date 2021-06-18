import cv2
import numpy as np


def summarize_video(video_path):
    """
    This function takes a video path as an input and returns a 6*5 matrix with concatenated images for every 
    minute of the video. 
    The image is also saved with the same name as the input video in the current working directory.
    """
    
    cap = cv2.VideoCapture(video_path)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    saved_frames = np.zeros([6*int(height), 5*int(width)])

    frame_no = 0
    saved_frame_count = 0

    while(frame_no < frame_length):   
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no);
        ret, frame = cap.read()
        frame_no += 60.0*fps

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            saved_frames[int(saved_frame_count/6)*int(height):(int(saved_frame_count/6)+1)*int(height), 
                       int(saved_frame_count%5)*int(width):(int(saved_frame_count%5)+1)*int(width)] = gray
            saved_frame_count += 1
        except Exception as e:
            print(e)
            break

    cap.release()

    img_name = '.'.join(video_path.split('/')[-1].split('.')[:-1]) + '.png'
    cv2.imwrite(img_name, saved_frames)
    return saved_frames

