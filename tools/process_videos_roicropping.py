import os
import glob
import subprocess
import cv2
import pandas as pd
import numpy as np
import argparse
import warnings
import tempfile
import sys
sys.path.append('../')
from utils.detect_roi import RoiDetector

# crop videos to labelled start and end of surgery
# resize to 256x256 to speed-up data loading and processing

def frame2minsec(frame, fps):
    seconds = frame/fps
    minutes = np.floor(seconds / 60).astype(int)
    seconds = np.floor(seconds % 60).astype(int)
    minsec = '{0:02d}:{1:02d}'.format(minutes, seconds)
    return minsec

def extract_frames_from_video(v_path, out_path, start, end, dim=256, fps=2.5):
    '''v_path: single video path;
       out_path: root to store output videos'''
    vname = os.path.splitext(os.path.basename(v_path))[0]
    vidcap = cv2.VideoCapture(v_path)
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

    if (width == 0) or (height==0):
        print(v_path, 'not successfully loaded, drop ..'); return

    start = '00:'+ start
    end = '00:' + end

    if not os.path.isdir(os.path.join(out_path, vname)):
        os.makedirs(os.path.join(out_path, vname))
    tmpfolder = tempfile.gettempdir()
    # crop video start to end
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error',
           '-i', '%s'% v_path,
           '-ss', start,
           '-to', end,
           '%s' % os.path.join(tmpfolder, vname + '.mp4')]
    ffmpeg = subprocess.call(cmd)

    # extract frames
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error',
           '-i', '%s' % os.path.join(tmpfolder, vname + '.mp4'),
           '-vf',
           'fps={0:.1f}'.format(fps),
           '%s' % os.path.join(out_path, vname, vname + '_%04d.png')]  # output path
    ffmpeg = subprocess.call(cmd)

    # glob all extracted frames and resize/crop with ROI
    detector = RoiDetector('../trained/hssstudy_intraop_unet_model_trained.pth')
    files = glob.glob(os.path.join(out_path, vname, '*.png'))
    assert len(files) > 0, 'no frames extracted'
    for file in files:
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        corners = detector(img)
        if corners is not None:
            img = img[corners[0][1]:corners[1][1], corners[0][0]:corners[1][0]]
        img = cv2.resize(img, (dim, dim))
        cv2.imwrite(file, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main_cataracts_train(input_path, output_path, fps=2.5):
    print('save to %s ... ' % output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    videos = sorted(glob.glob(os.path.join(input_path, '*.mp4')))
    assert len(videos) > 0, f'no mp4 videos found in {input_path}'
    labels = sorted(glob.glob(os.path.join(input_path, '*.csv')))
    assert len(videos) == len(labels), 'not same number of videos and labels found'
    for i, (vid, lab) in enumerate(zip(videos, labels)):
        print(f'{vid}: {i+1} of {len(videos)}')
        # sample label file
        vname = os.path.splitext(os.path.basename(vid))[0]
        if os.path.isfile(os.path.join(output_path, vname + '.mp4')):
            continue
        label = pd.read_csv(lab)['Steps'].to_numpy()
        label = np.concatenate((label, [0]))  # make sure the last value is 0 to find the edge
        start_frame = np.min(np.where((np.diff(label, prepend=0) != 0) & (label == 3))[0])
        end_frame = np.max(np.where((np.diff(label, prepend=0) != 0) & (label == 0))[0])
        dummy_reader = cv2.VideoCapture(vid)
        orig_fps = dummy_reader.get(cv2.CAP_PROP_FPS)
        start = frame2minsec(start_frame, orig_fps)
        end = frame2minsec(end_frame, orig_fps)
        extract_frames_from_video(vid, output_path, dim=256, fps=fps, start=start, end=end)


def main_others(input_path, output_path, fps=2.5):
    print('save to %s ... ' % output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    videos = sorted(glob.glob(os.path.join(input_path, '*.mp4')))
    assert len(videos) > 0, f'no mp4 videos found in {input_path}'
    assert os.path.isfile(os.path.join(input_path, 'start_end_labels.csv')), 'start end label file not found'
    labels = pd.read_csv(os.path.join(input_path, 'start_end_labels.csv'))
    for i, vid in enumerate(videos):
        print(f'{vid}: {i+1} of {len(videos)}')
        # sample label file
        vname = os.path.splitext(os.path.basename(vid))[0]
        if os.path.isfile(os.path.join(output_path, vname + '.mp4')):
            continue
        p_label = labels[labels.PatientID.eq(vname)]
        assert len(p_label) == 1, f'no or multiple entries for {vname}'
        extract_frames_from_video(vid, output_path, dim=256, fps=fps, start=p_label['Start'].values[0], end=p_label['End'].values[0])


def main_cataract101(input_path, output_path, fps=2.5):
    print('save to %s ... ' % output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    videos = sorted(glob.glob(os.path.join(input_path, '*.mp4')))
    assert len(videos) > 0, f'no mp4 videos found in {input_path}'
    labels = sorted(glob.glob(os.path.join(input_path, '*.csv')))
    assert len(videos) == len(labels), 'not same number of videos and labels found'
    for i, (vid, lab) in enumerate(zip(videos, labels)):
        print(f'{vid}: {i + 1} of {len(videos)}')
        # sample label file
        vname = os.path.splitext(os.path.basename(vid))[0]
        vidcap = cv2.VideoCapture(vid)
        orig_fps = vidcap.get(cv2.CAP_PROP_FPS)  # float
        sample_factor = np.floor(orig_fps/fps)
        if sample_factor != (orig_fps/fps):
            fps = orig_fps/sample_factor
            warnings.warn('original fps cannot by divided by new fps, use {0:.2f} instead'.format(fps))
        label_df = pd.read_csv(lab)
        label = np.concatenate((label_df['valid'].to_numpy(), [0]))  # make sure the last value is 0 to find the edge
        start_frame = np.min(np.where((np.diff(label, prepend=0) != 0) & (label == 1))[0])
        end_frame = np.max(np.where((np.diff(label, prepend=0) != 0) & (label == 0))[0])
        dummy_reader = cv2.VideoCapture(vid)
        orig_fps = dummy_reader.get(cv2.CAP_PROP_FPS)
        start = frame2minsec(start_frame, orig_fps)
        end = frame2minsec(end_frame, orig_fps)
        label_df = label_df[start_frame:end_frame]
        label_df = label_df[::int(sample_factor)]
        if not os.path.isdir(os.path.join(output_path, vname)):
            os.makedirs(os.path.join(output_path, vname))
            label_df.to_csv(os.path.join(output_path, vname, os.path.basename(lab)))
            extract_frames_from_video(vid, output_path, dim=256, fps=fps, start=start, end=end)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        help='path to folder containing video files.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../output',
        help='path to output folder.'
    )
    parser.add_argument(
        '--label',
        type=str,
        choices=['cataract101', 'CATARACTs', 'startend'],
        default='startend',
        help='type of label, if startend provide a csv file with fields {PatientID,Start,End}. '
             'Start/End in format XX:XX.'
    )
    args = parser.parse_args()
    if args.label == 'cataract101':
        main_cataract101(args.input, args.output)
    elif args.label == 'CATARACTs':
        main_cataracts_train(args.input, args.output)
    else:
        main_others(args.input, args.output)
