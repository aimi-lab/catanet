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
import pickle
import pandas as pd


def evaluate_test(model, features, rsd_labels, device):
    model.eval()
    X = torch.tensor(features).unsqueeze(0).to(device)
    with torch.no_grad():
        rsd_prediction = model.forwardRNN(X, skip_features=True).squeeze().cpu().numpy()
    diff = rsd_labels - rsd_prediction
    mae_all = np.mean(np.abs(diff))
    mae_5 = np.mean(np.abs(diff[rsd_labels < 5]))
    return mae_all, mae_5


def main(output_folder, log, pretrained_model, config, runs=10):
    config['train'] = {}
    config['val'] = {}
    config["train"]['batch_size'] = 1
    config["train"]['epochs'] = [50]
    config["train"]["learning_rate"] = [0.0001]
    config['train']['weighted_loss'] = True
    config["val"]['batch_size'] = 1
    config["pretrained_model"] = pretrained_model
    config["train"]["sequence"] = ['train_rnn']
    config['train']['window'] = 48
    config['input_size'] = 224
    config['val']['test_every'] = 1

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #specify if we should use a GPU (cuda) or only the CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- training params
    training_phases = ['train', 'val']

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

    # precompute CNN features on test set
    test_sequences = {'basel': sorted(glob.glob('../data/basel/test/*/')),
                      'cataract101':sorted(glob.glob('../data/cataract101/test/*/'))}

    all_df = []
    for counter, subset in enumerate([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]):
        print(f'selecting {100 * subset}% random subset')
        for run_id in range(runs):
            # --- model
            model = CatRSDNet_NL()
            model.load_catrsdnet(torch.load(config['pretrained_model'])['model_dict'])
            model.set_cnn_as_feature_extractor()
            model = model.to(device)
            # --- glob data set
            sequences_path = {key: {} for key in training_phases}
            sequences_path['train']['video'] = sorted(glob.glob(os.path.join(config['data']['base_path'], 'train', '*/')))
            sequences_path['val']['video'] = sorted(glob.glob(os.path.join(config['data']['base_path'], 'val', '*/')))
            # randomly select subset of videos
            random.shuffle(sequences_path['train']['video'])
            random.shuffle(sequences_path['val']['video'])
            N_train = int(np.round(len(sequences_path['train']['video'])*subset))
            N_val = int(np.round(len(sequences_path['val']['video']) * subset))
            sequences_path['train']['video'] = sequences_path['train']['video'][:N_train]
            sequences_path['val']['video'] = sequences_path['val']['video'][:N_val]

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
                        if os.path.isfile('precomputed_features3.pckl'):
                            with open('precomputed_features3.pckl', 'rb') as f:
                                features, rsd_labels = pickle.load(f)[:2]
                        else:
                            model.eval()
                            sequences = sequences_path['train']['video']+sequences_path['val']['video']+\
                                        test_sequences['basel']+test_sequences['cataract101']
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
                            with open('precomputed_features3.pckl', 'wb') as f:
                                pickle.dump([features, rsd_labels], f)
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

            # evaluate on test set
            model.load_state_dict(torch.load(output_model_name)['model_dict'])
            for testset in test_sequences:
                for f in test_sequences[testset]:
                    mae, mae5 = evaluate_test(model, features[f], rsd_labels[f], device)
                    all_df.append([testset, subset, f, mae, mae5])

    all_df = pd.DataFrame(data=all_df, columns=['dataset', 'subset', 'filename', 'mae', 'mae5'])
    all_df.to_csv(os.path.join(output_folder, 'all_run_summary.csv'))
    print(all_df.groupby(['dataset', 'subset'])['mae'].mean())
    print(all_df.groupby(['dataset', 'subset'])['mae'].std())
    print(all_df.groupby(['dataset', 'subset'])['mae5'].mean())
    print(all_df.groupby(['dataset', 'subset'])['mae5'].std())

    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams['font.size'] = 16
    sns.lineplot(x='subset', y='mae', data=all_df, hue='dataset')
    plt.xlabel('Proportion of Basel Data')
    plt.ylabel('RSD Mean Absolute Error')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'mae.pdf'))
    plt.close()

    sns.lineplot(x='subset', y='mae5', data=all_df, hue='dataset')
    plt.xlabel('Proportion of Basel Data')
    plt.ylabel('RSD Mean Absolute Error 5 min')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'mae_5.pdf'))


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
