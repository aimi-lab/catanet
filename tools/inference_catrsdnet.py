import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import Compose, ToTensor, ToPILImage, Resize
import sys
sys.path.append('../')
from models.catRSDNet import CatRSDNet
from models.catRSDNet_NL import CatRSDNet_NL
import glob
from utils.dataset_utils import DatasetNoLabel


def main(out, input, checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- model
    assert os.path.isfile(checkpoint), checkpoint + ' is not a file.'

    # check which model is used CatNet or CatNet_NL
    checkpoint = torch.load(checkpoint, map_location='cpu')
    try:
        model = CatRSDNet().to(device)
        model.set_cnn_as_feature_extractor()
        model.load_state_dict(checkpoint['model_dict'])
        model_type = 'CatNet'
    except RuntimeError:
        model = CatRSDNet_NL().to(device)
        model.set_cnn_as_feature_extractor()
        model.load_state_dict(checkpoint['model_dict'])
        model_type = 'CatNet_NL'
    print(model_type, ' loaded')
    model.eval()

    model = model.to(device)
    # find input format
    assert os.path.isdir(input), 'no valid input provided, needs to be a folder, run the process_video.py file first'
    video_folders = sorted(glob.glob(os.path.join(input, '*/')))
    if len(video_folders) == 0:
        video_folders = [input]

    for file in video_folders:
        vname = os.path.basename(os.path.dirname(file))
        # compute cnn features for whole video
        data = DatasetNoLabel(datafolders=[file], img_transform=Compose([ToPILImage(), Resize(224), ToTensor()]))
        dataloader = DataLoader(data, batch_size=200, shuffle=False, num_workers=1, pin_memory=True)
        outputs = []
        print('start inference on ', vname)
        for ii, (X, elapsed_time, frame_no, rsd) in enumerate(dataloader):  # for each of the batches
            X = X.to(device)
            elapsed_time = elapsed_time.unsqueeze(0).float().to(device)
            with torch.no_grad():
                if model_type == 'CatNet':
                    step_pred, exp_pred, rsd_pred = model(X, stateful=(ii > 0))
                    step_pred_hard = torch.argmax(step_pred, dim=-1).view(-1).cpu().numpy()
                    exp_pred_soft = exp_pred.clone().cpu().numpy()
                    exp_pred = (torch.argmax(exp_pred, dim=-1) + 1).view(-1).cpu().numpy()
                else:  # CatNet-NL
                    rsd_pred = model(X, stateful=(ii > 0))
                    step_pred_hard = np.zeros(len(rsd_pred))  # dummy
                    exp_pred = np.zeros(len(rsd_pred))  # dummy
                    exp_pred_soft = np.zeros((len(rsd_pred), 2))  # dummy
                progress_pred = elapsed_time/(elapsed_time+rsd_pred.T+0.00001)
                progress_pred = progress_pred.view(-1).cpu().numpy()
                rsd_pred = rsd_pred.view(-1).cpu().numpy()
                elapsed_time = elapsed_time.view(-1).cpu().numpy()
            outputs.append(np.asarray([elapsed_time, progress_pred, rsd_pred, step_pred_hard, exp_pred, exp_pred_soft[:,0], exp_pred_soft[:,1]]).T)
        outputs = np.concatenate(outputs)
        np.savetxt(os.path.join(out, f'{vname}.csv'), outputs, delimiter=',',
                   header='elapsed,progress,predicted_rsd,predicted_step,predicted_exp,predicted_assistant,predicted_senior', comments='')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--out',
        type=str,
        default='output',
        help='path to output folder.'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='path to processed video file or multiple files.'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='path to to model checkpoint .pth file.'
    )

    args = parser.parse_args()

    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    main(out=args.out, input=args.input, checkpoint=args.checkpoint)

