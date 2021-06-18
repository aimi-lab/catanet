import numpy as np
import argparse
import pandas as pd
import cv2
import os
import glob


def overlay_phase(path_to_video, output_path, class_labels,start, end, gt=None, prediction=None, sampling_factor=1, size=3):
    assert os.path.isfile(path_to_video), 'no valid video file {0}'.format(path_to_video)
    reader = cv2.VideoCapture(path_to_video)
    fps = int(reader.get(cv2.CAP_PROP_FPS))
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    i = 0
    j = 0
    while True:
        ret, frame = reader.read()
        if not ret:
            break
        j += 1
        if j < start:
            continue
        if j > end:
            break
        if gt is not None:
            overlay = frame.copy()
            bg_color = (0, 0, 0)
            bg = np.full((frame.shape), bg_color, dtype=np.uint8)
            cv2.putText(bg, 'ground-truth: ' + class_labels[gt[i]], (10, 20*size), cv2.FONT_HERSHEY_SIMPLEX, size,
                        [0, 0, 255])
            x, y, w, h = cv2.boundingRect(bg[:, :, 2])
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 1 - 0.6, 0, frame)
            cv2.putText(frame, 'ground-truth: ' + class_labels[gt[i]], (10, 20*size), cv2.FONT_HERSHEY_SIMPLEX, size, [255, 255, 255])
        if prediction is not None:
            overlay = frame.copy()
            bg_color = (0, 0, 0)
            bg = np.full((frame.shape), bg_color, dtype=np.uint8)
            cv2.putText(bg, 'predicted: ' + class_labels[prediction[i]], (10, 50*size), cv2.FONT_HERSHEY_SIMPLEX, size,
                        [0, 0, 255])
            x, y, w, h = cv2.boundingRect(bg[:, :, 2])
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 1 - 0.5, 0, frame)
            if gt is not None:
                if prediction[i] == gt[i]:
                    cv2.putText(frame, 'predicted: ' + class_labels[prediction[i]], (10, 50*size), cv2.FONT_HERSHEY_SIMPLEX, size, [0, 255, 0])
                else:
                    cv2.putText(frame, 'predicted: ' + class_labels[prediction[i]], (10, 50*size), cv2.FONT_HERSHEY_SIMPLEX, size,
                            [0, 0, 255])
            else:
                cv2.putText(frame, 'predicted: ' + class_labels[prediction[i]], (10, 50*size), cv2.FONT_HERSHEY_SIMPLEX, size,
                            [255, 0, 0])
        writer.write(frame)
        if (j%sampling_factor) == 0:
            i += 1
    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--out',
        type=str,
        default='output',
        help='path to output folder.'
    )
    parser.add_argument(
        '--video',
        type=str,
        help='path to folder with videos or single video file.'
    )
    parser.add_argument(
        '--prediction',
        type=str,
        help='path to folder with inference csv files or single file.'
    )

    args = parser.parse_args()

    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    class_labels = ['None','Incision', 'VisAgInj', 'Rhexis', 'Hydrodissection', 'Phaco', 'IrAsp', 'CapsPol', 'LensImpl',
                    'VisAgRem', 'TonAnti']

    video_paths = sorted(glob.glob(args.video))
    prediction_path = sorted(glob.glob(args.prediction))

    for v, p in zip(video_paths, prediction_path):
        fname = os.path.splitext(os.path.basename(p))[0]
        idx = os.path.basename(v)
        print(idx)
        output_path = os.path.join('/home/mhayoz/Desktop', idx)
        pred_df = pd.read_csv(p, index_col=0)
        predictions = pred_df['predicted_step'].to_numpy().astype(int)
        labels = pred_df['gt_step'].to_numpy().astype(int)
        start = int(pred_df.index[0])
        end = int(pred_df.index[-1])
        overlay_phase(v, output_path, class_labels, prediction=predictions, gt=labels, size=1, start=start, end=end,
                      sampling_factor=10)


