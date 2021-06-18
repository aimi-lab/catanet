import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
import sys
import yaml
import time
import wandb
import shutil
import tempfile
sys.path.append('../')
from models.catRSDNet_NL import CatRSDNet_NL
from utils.dataset_utils import DatasetNoLabel
from utils.logging_utils import timeSince
import glob
from torchvision.transforms import Compose, RandomResizedCrop, RandomVerticalFlip, RandomHorizontalFlip, ToPILImage, \
    ToTensor, Resize


def main(output_folder, log, config, pretrained):
    # specify videos of surgeons for training and validation
    config['train'] = {}
    config['val'] = {}
    config["train"]['batch_size'] = 100
    config["train"]['epochs'] = 2
    config['train']['sub_epoch_validation'] = 50
    config['train']['learning_rate'] = 0.00001
    config["val"]['batch_size'] = 150
    config['input_size'] = [224, 224]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # specify if we should use a GPU (cuda) or only the CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_devices = torch.cuda.device_count()

    # --- training params
    training_phases = ['train', 'val']
    img_transform = {}
    img_transform['train'] = Compose([ToPILImage(), RandomHorizontalFlip(), RandomVerticalFlip(),
                                      RandomResizedCrop(size=config['input_size'][0], scale=(0.4,1.0), ratio=(1.0,1.0)),
                                      ToTensor()])
    img_transform['val'] = Compose([ToPILImage(), Resize(config['input_size']), ToTensor()])

    # --- logging
    if log:
        run = wandb.init(project='cataract_rsd', group='catnet_noexp')
        run.config.data = config['data']['base_path']
        run.name = run.id
    # --- glob data set
    # copy datafolders to scratch - transferring images seems to be a bottleneck
    tmp_basepath = os.path.join(tempfile.gettempdir(), 'data')
    if not os.path.isdir(tmp_basepath):
        print('copy data to ', tmp_basepath, '...')
        shutil.copytree(config['data']['base_path'], tmp_basepath)
    dataLoader = {}
    for phase in training_phases:
        data_folders = sorted(glob.glob(os.path.join(tmp_basepath, phase, '*')))
        dataset = DatasetNoLabel(data_folders, img_transform=img_transform[phase])
        dataLoader[phase] = DataLoader(dataset, batch_size=config[phase]['batch_size'],
                                       shuffle=(phase == 'train'), num_workers=4, pin_memory=True)

    output_model_name = os.path.join(output_folder, 'catRSDNet_CNN.pth')

    print('start training... ')
    # --- model
    base_model = CatRSDNet_NL()
    base_model.load_catrsdnet(torch.load(pretrained)['model_dict'])
    model = base_model.cnn
    # we have a different training objective when training without labels. So we train the last layer from scratch.
    model.classifier = torch.nn.Linear(1664, 1)
    model = model.to(device)

    if num_devices > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    # --- optimizer
    optim = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    # --- loss
    criterion = nn.L1Loss()

    # --- training
    best_loss_on_test = np.Infinity
    start_time = time.time()
    stop_epoch = config['train']['epochs']

    for epoch in range(stop_epoch):
        #zero out epoch based performance variables
        all_loss_train = torch.zeros(0).to(device)
        model.train()  # Set model to training mode

        for ii, (img, elapsed_time, frame_number, rsd) in enumerate(dataLoader['train']):
            img = img.to(device)  # input data
            with torch.set_grad_enabled(True):
                rsd_prediction = model(img)
                loss = criterion(rsd_prediction.squeeze(1), rsd.float().to(device))
                # update weights
                optim.zero_grad()
                loss.backward()
                optim.step()

            all_loss_train = torch.cat((all_loss_train, loss.detach().view(1, -1)))

            # compute sub-epoch validation loss for early stopping
            if ii % config['train']['sub_epoch_validation'] == 0:
                model.eval()
                with torch.no_grad():
                    val_subepoch_loss = torch.zeros(0).to(device)
                    for jj, (img, elapsed_time, frame_number, rsd) in enumerate(dataLoader['val']):
                        img = img.to(device)  # input data
                        rsd_prediction = model(img)
                        loss = criterion(rsd_prediction.squeeze(1), rsd.float().to(device))
                        val_subepoch_loss = torch.cat((val_subepoch_loss, loss.detach().view(1, -1)))
                # compute metrics
                val_subepoch_loss = val_subepoch_loss.cpu().numpy().mean()
                print('val loss: {0:.4f}'.format(val_subepoch_loss), end='')
                if log:
                    wandb.log({'/val/loss': val_subepoch_loss})

                if val_subepoch_loss < best_loss_on_test:
                    # if current loss is the best we've seen, save model state
                    if num_devices > 1:
                        state_dict = model.module.state_dict()
                    else:
                        state_dict = model.state_dict()
                    best_loss_on_test = val_subepoch_loss
                    print('  **')
                    state = {'epoch': epoch + 1,
                             'model_dict': state_dict
                             }

                    torch.save(state, output_model_name)
                else:
                    print('')
                model.train()

        all_loss_train = all_loss_train.cpu().numpy().mean()
        if log:
            wandb.log({'epoch': epoch, '/train/loss': all_loss_train})

        log_text = '%s ([%d/%d] %d%%), train loss: %.4f' %\
                   (timeSince(start_time, (epoch+1) / stop_epoch),
                    epoch + 1, stop_epoch , (epoch + 1) / stop_epoch * 100,
                    all_loss_train)
        print(log_text)

    print('...finished training')


if __name__ == "__main__":
    """The program's entry point."""
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Training CatNet CNN without labels')

    parser.add_argument(
        '--out',
        type=str,
        default='output',
        help='Path to output file, ignored if log is true (use wandb directory instead).'
    )
    parser.add_argument(
        '--log',
        type=str2bool,
        default='False',
        help='if true log with wandb.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../configuration/config_basel.yaml',
        help='path to config file.'
    )
    parser.add_argument(
        '--pretrained',
        type=str,
        help='path to pre-trained CNN.'
    )
    args = parser.parse_args()

    assert os.path.isfile(args.config), f'{args.config} is not a file'
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    main(output_folder=args.out, log=args.log, config=config, pretrained=args.pretrained)
