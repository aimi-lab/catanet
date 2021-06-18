import torch
from torch import nn
import os
import argparse
import numpy as np
import sys
import yaml
import time
import wandb
from torch.utils.data import DataLoader
import glob
sys.path.append('../')
from models.catRSDNet_NL import CatRSDNet_NL
from utils.dataset_utils import DatasetNoLabel
from utils.logging_utils import timeSince
from torchvision import transforms
import random
import warnings


def conf2metrics(conf_mat):
    """ Confusion matrix to performance metrics conversion """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = np.diag(conf_mat) / np.sum(conf_mat, axis=0)
    precision[np.isnan(precision)] = 0.0

    recall = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
    recall[np.isnan(recall)] = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1 = 2 * (precision * recall) / (precision + recall)
    f1[np.isnan(f1)] = 0.0

    accuracy = np.trace(conf_mat) / np.sum(conf_mat)
    return precision, recall, f1, accuracy


def main(output_folder, log, pretrained_model, config):
    config['train'] = {}
    config['val'] = {}
    config["train"]['batch_size'] = 1
    config["train"]['epochs'] = [50, 10, 20]
    config["train"]["learning_rate"] = [0.001, 0.0001, 0.0005]
    config['train']['weighted_loss'] = True
    config["val"]['batch_size'] = 1
    config["pretrained_model"] = pretrained_model
    config["train"]["sequence"] = ['train_rnn', 'train_all', 'train_rnn']
    config['train']['window'] = 48
    config['input_size'] = 224
    config['val']['test_every'] = 1

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #specify if we should use a GPU (cuda) or only the CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_devices = torch.cuda.device_count()

    # --- model
    model = CatRSDNet_NL()
    model.cnn.load_state_dict(torch.load(config['pretrained_model'])['model_dict'])
    model.set_cnn_as_feature_extractor()
    model = model.to(device)

    # --- training params
    training_phases = ['train', 'val']
    validation_phases = ['val']

    # --- logging
    if log:
        run = wandb.init(project='cataract_rsd', group='catnet_noexp')
        run.config.data = config['data']['base_path']
        run.name = run.id

    output_model_name = os.path.join(output_folder, 'catRSDNet.pth')

    # --- pre-processing data
    print('collect dataset')
    # glob all video files and perform forward pass through CNN. Store the features in npy-arrays for RNN training
    img_transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(config['input_size']),
                                        transforms.ToTensor()])
    # --- glob data set
    sequences_path = {key:{} for key in training_phases}
    sequences_path['train']['video'] = sorted(glob.glob(os.path.join(config['data']['base_path'], 'train', '*/')))
    sequences_path['val']['video'] = sorted(glob.glob(os.path.join(config['data']['base_path'], 'val', '*/')))

    print('number of sequences: train {0}, val {1}'.format(len(sequences_path['train']['video']),
                                                           len(sequences_path['val']['video'])))
    # --- Loss and Optimizer
    criterion = nn.L1Loss()

    # --- training
    # training steps:
    training_steps = config['train']['sequence']

    remaining_steps = training_steps.copy()
    print(' start training... ')

    start_time = time.time()
    non_improving_val_counter = 0
    features = {}
    rsd_labels = {}
    for step_count, training_step in enumerate(training_steps):
        print(training_step)
        if step_count > 0:
            checkpoint = torch.load(output_model_name, map_location=device)
            model.load_state_dict(checkpoint['model_dict'])
        best_loss_on_val = np.Infinity
        stop_epoch = config['train']['epochs'][step_count]
        # optimizer
        optim = torch.optim.Adam([{'params': model.rnn.parameters()},
                                  {'params': model.cnn.parameters(), 'lr': config['train']['learning_rate'][step_count] / 20}],
                                  lr=config['train']['learning_rate'][step_count])

        if training_step == 'train_rnn':
            # pre-compute features
            if len(features) == 0:
                model.eval()
                sequences = sequences_path['train']['video']+sequences_path['val']['video']
                for ii, input_path in enumerate(sequences):
                    print(input_path)
                    data = DatasetNoLabel([input_path], img_transform=img_transform)
                    dataloader = DataLoader(data, batch_size=500, shuffle=False, num_workers=1, pin_memory=True)
                    features[input_path] = []
                    rsd_labels[input_path] = []
                    for i, (X, _, _, y_rsd) in enumerate(dataloader):
                        with torch.no_grad():
                            features[input_path].append(model.cnn(X.float().to(device)).cpu().numpy())
                            rsd_labels[input_path].append(y_rsd.numpy())
                    features[input_path] = np.concatenate(features[input_path])
                    rsd_labels[input_path] = np.concatenate(rsd_labels[input_path])
            model.freeze_cnn(True)
            model.freeze_rnn(False)
        elif training_step == 'train_all':
            model.freeze_cnn(False)
            model.freeze_rnn(False)
            features = {}
            rsd_labels = {}
        else:
            raise RuntimeError('training step {0} not implemented'.format(training_step))

        for epoch in range(stop_epoch):
            #zero out epoch based performance variables
            all_loss = {key: torch.zeros(0).to(device) for key in training_phases}

            for phase in training_phases: #iterate through both training and validation states
                if phase == 'train':
                    model.train()  # Set model to training mode
                    random.shuffle(sequences_path[phase])  # random shuffle training sequences
                    model.cnn.eval()  # required due to batch-norm, even when training end-to-end
                else:
                    model.eval()   # Set model to evaluate mode

                for ii,  input_path in enumerate(sequences_path[phase]['video']):
                    if (training_step == 'train_rnn') | (training_step == 'train_fc'):
                        dataloader = [(torch.tensor(features[input_path]).unsqueeze(0),
                                       None, None, torch.tensor(rsd_labels[input_path]).unsqueeze(0))]
                        skip_features = True
                    else:
                        batch_size = config['train']['window']
                        data = DatasetNoLabel([input_path], img_transform=img_transform)
                        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
                        skip_features = False

                    for i, (X, _, _, y_rsd) in enumerate(dataloader):
                        if len(y_rsd.shape) == 2:
                            y_rsd = y_rsd.squeeze(0)
                        y_rsd = y_rsd.float().to(device)
                        X = X.float().to(device)

                        with torch.set_grad_enabled(phase == 'train'):
                            stateful = (i > 0)
                            rsd_prediction = model.forwardRNN(X, stateful=stateful, skip_features=skip_features)
                            loss = criterion(rsd_prediction.squeeze(1), y_rsd)
                            if phase == 'train':  # in case we're in train mode, need to do back propagation
                                optim.zero_grad()
                                loss.backward()
                                optim.step()

                            all_loss[phase] = torch.cat((all_loss[phase], loss.detach().view(1, -1)))

                all_loss[phase] = all_loss[phase].cpu().numpy().mean()

                if log:
                    log_epoch = step_count*epoch+epoch
                    wandb.log({'epoch': log_epoch, f'{phase}/loss': all_loss[phase],
                               f'{phase}/loss_rsd': all_loss[phase]})

            log_text = '%s ([%d/%d] %d%%), train loss: %.4f val loss: %.4f' % \
                       (timeSince(start_time, (epoch + 1) / stop_epoch),
                        epoch + 1, stop_epoch, (epoch + 1) / stop_epoch * 100,
                        all_loss['train'], all_loss['val'])
            print(log_text, end='')

            # if current loss is the best we've seen, save model state
            if all_loss["val"] < best_loss_on_val:
                # if current loss is the best we've seen, save model state
                non_improving_val_counter = 0
                best_loss_on_val = all_loss["val"]
                print('  **')
                state = {'epoch': epoch + 1,
                         'model_dict': model.state_dict(),
                         'remaining_steps': remaining_steps}

                torch.save(state, output_model_name)
                if log:
                    wandb.summary['best_epoch'] = epoch + 1
                    wandb.summary['best_loss_on_val'] = best_loss_on_val

            else:
                print('')
                non_improving_val_counter += 1
        remaining_steps.pop(0)
    print('...finished training ')


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

    parser = argparse.ArgumentParser(description='Training RNN for Cataract Tool Detection')

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
        '--pretrained',
        type=str,
        help='path to pre-trained CNN for CatNet.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../configuration/config_basel.yaml',
        help='path to config file.'
    )
    args = parser.parse_args()

    assert os.path.isfile(args.config), f'{args.config} is not a file'
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    main(output_folder=args.out, log=args.log, pretrained_model=args.pretrained, config=config)
