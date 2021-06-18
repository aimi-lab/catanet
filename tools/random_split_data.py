import argparse
import numpy as np
import os
import glob
import shutil


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        help='path to folder containing processed video-folders.'
    )
    parser.add_argument(
        '--out',
        type=str,
        default='../output',
        help='path to output folder.'
    )
    parser.add_argument(
        '--split',
        type=list,
        default=[70, 20, 10],
        help='List of proportions for train, val, test. Must sum to 100.'
    )
    args = parser.parse_args()
    assert np.sum(args.split) == 100, 'split must sum to one'
    split = np.asarray(args.split)/100.0

    videos = glob.glob(os.path.join(args.input, '*/'))
    assert len(videos) > 0, f'no video folders found in {args.input}'

    if not os.path.isdir(args.out):
        [os.makedirs(os.path.join(args.out, phase)) for phase in ['train', 'val', 'test']]

    ids = np.random.permutation(np.arange(len(videos)))
    ids_phase = {}
    ids_phase['train'] = ids[:np.floor(split[0]*len(videos)).astype(int)]
    ids_phase['val'] = ids[np.floor(split[0] * len(videos)).astype(int):
                  np.floor((split[0]+split[1]) * len(videos)).astype(int)]
    ids_phase['test'] = ids[np.floor((split[0]+split[1]) * len(videos)).astype(int):]
    for phase in ids_phase:
        for id in ids_phase[phase]:
            video_b = os.path.basename(os.path.dirname(videos[id]))
            shutil.move(videos[id], os.path.join(args.out, phase, video_b))

