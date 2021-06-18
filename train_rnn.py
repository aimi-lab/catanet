import torch
from torch import nn
import os
import argparse
import numpy as np
import sys
import time
import wandb
from torch.utils.data import DataLoader
import glob
from models.catRSDNet import CatRSDNet
from utils.dataset_utils import DatasetCataract101
from utils.logging_utils import timeSince
from torchvision import transforms
import csv
from sklearn.metrics import confusion_matrix
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


def main(output_folder, log, pretrained_model):
    config = {'train': {}, 'val': {}, 'data': {}}
    config["train"]['batch_size'] = 1
    config["train"]['epochs'] = [50, 10, 20]
    config["train"]["learning_rate"] = [0.001, 0.0001, 0.0005]
    config['train']['weighted_loss'] = True
    config["val"]['batch_size'] = 1
    config["pretrained_model"] = pretrained_model
    config["data"]["base_path"] = 'data/cataract101'
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
    n_step_classes = 11
    model = CatRSDNet()
    model.cnn.load_state_dict(torch.load(config['pretrained_model'])['model_dict'])
    model.set_cnn_as_feature_extractor()
    model = model.to(device)

    # --- training params
    training_phases = ['train', 'val']
    validation_phases = ['val']

    # --- logging
    if log:
        run = wandb.init(project='cataract_rsd', group='catnet')
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
    sequences_path['train']['label'] = sorted(glob.glob(os.path.join(config['data']['base_path'], 'train','**','*.csv')))
    sequences_path['val']['label'] = sorted(glob.glob(os.path.join(config['data']['base_path'], 'val', '**', '*.csv')))
    sequences_path['train']['video'] = sorted(glob.glob(os.path.join(config['data']['base_path'], 'train', '*/')))
    sequences_path['val']['video'] = sorted(glob.glob(os.path.join(config['data']['base_path'], 'val', '*/')))

    print('number of sequences: train {0}, val {1}'.format(len(sequences_path['train']['label']),
                                                           len(sequences_path['val']['label'])))
    # --- Loss and Optimizer
    # loss function
    if config['train']['weighted_loss']:
        label_sum = np.zeros(n_step_classes)
        for fname_label in sequences_path['train']['label']:
            labels = np.genfromtxt(fname_label, delimiter=',', skip_header=1)[:, 1]
            for l in range(n_step_classes):
                label_sum[l] += np.sum(labels==l)
        loss_weights = 1 / label_sum
        loss_weights[label_sum == 0] = 0.0
        loss_weights = torch.tensor(loss_weights / np.max(loss_weights)).float().to(device)
    else:
        loss_weights = None

    step_criterion = nn.CrossEntropyLoss(weight=loss_weights)
    experience_criterion = nn.CrossEntropyLoss()
    rsd_criterion = nn.L1Loss()

    # --- training
    # training steps:
    training_steps = config['train']['sequence']

    remaining_steps = training_steps.copy()
    print(' start training... ')

    start_time = time.time()
    non_improving_val_counter = 0
    features = {}
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
                sequences = list(zip(sequences_path['train']['label']+sequences_path['val']['label'],
                                     sequences_path['train']['video']+sequences_path['val']['video']))
                for ii, (label_path, input_path) in enumerate(sequences):
                    print(input_path)
                    data = DatasetCataract101([input_path], [label_path], img_transform=img_transform)
                    dataloader = DataLoader(data, batch_size=500, shuffle=False, num_workers=1, pin_memory=True)
                    features[input_path] = []
                    for i, (X, _) in enumerate(dataloader):
                        with torch.no_grad():
                            features[input_path].append(model.cnn(X.float().to(device)).cpu().numpy())
                    features[input_path] = np.concatenate(features[input_path])
            model.freeze_cnn(True)
            model.freeze_rnn(False)
        elif training_step == 'train_all':
            model.freeze_cnn(False)
            model.freeze_rnn(False)
            features = {}
        else:
            raise RuntimeError('training step {0} not implemented'.format(training_step))

        for epoch in range(stop_epoch):
            #zero out epoch based performance variables
            all_precision = {}
            average_precision = {}
            all_recall = {}
            average_recall = {}
            all_f1 = {}
            average_f1 = {}
            accuracy_exp = {}
            conf_mat = {key: np.zeros((n_step_classes, n_step_classes)) for key in validation_phases}
            conf_mat_exp = {key: np.zeros((2, 2)) for key in validation_phases}
            all_loss = {key: torch.zeros(0).to(device) for key in training_phases}
            all_loss_step = {key: torch.zeros(0).to(device) for key in training_phases}
            all_loss_experience = {key: torch.zeros(0).to(device) for key in training_phases}
            all_loss_rsd = {key: torch.zeros(0).to(device) for key in training_phases}

            for phase in training_phases: #iterate through both training and validation states
                sequences = list(zip(sequences_path[phase]['label'], sequences_path[phase]['video']))
                if phase == 'train':
                    model.train()  # Set model to training mode
                    random.shuffle(sequences)  # random shuffle training sequences
                    model.cnn.eval()  # required due to batch-norm, even when training end-to-end
                else:
                    model.eval()   # Set model to evaluate mode

                for ii, (label_path, input_path) in enumerate(sequences):
                    if (training_step == 'train_rnn') | (training_step == 'train_fc'):
                        label = torch.tensor(np.genfromtxt(label_path, delimiter=',', skip_header=1)[:, 1:])
                        dataloader = [(torch.tensor(features[input_path]).unsqueeze(0),
                                      label[:len(features[input_path]),:])]
                        skip_features = True
                    else:
                        batch_size = config['train']['window']
                        data = DatasetCataract101([input_path], [label_path], img_transform=img_transform)
                        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
                        skip_features = False

                    for i, (X, y) in enumerate(dataloader):
                        if len(y.shape) > 2: # batch-size is automatically removed from tensor dimensions for label
                            y = y.squeeze()
                        y_experience = torch.add(y[:, 2], -1).long().to(device)
                        y_rsd = (y[:, 5]/60.0/25.0).float().to(device)
                        y = y[:, 0].long().to(device)
                        X = X.float().to(device)

                        with torch.set_grad_enabled(phase == 'train'):
                            stateful = (i > 0)
                            step_prediction, experience_prediction, rsd_prediction = model.forwardRNN(X, stateful=stateful,
                                                                                      skip_features=skip_features)
                            loss_step = step_criterion(step_prediction, y)
                            loss_experience = experience_criterion(experience_prediction, y_experience)
                            rsd_prediction = rsd_prediction.squeeze(1)
                            loss_rsd = rsd_criterion(rsd_prediction, y_rsd)
                            loss = loss_step + 0.3 * loss_experience + loss_rsd
                            if phase == 'train':  # in case we're in train mode, need to do back propagation
                                optim.zero_grad()
                                loss.backward()
                                optim.step()

                            all_loss[phase] = torch.cat((all_loss[phase], loss.detach().view(1, -1)))
                            all_loss_step[phase] = torch.cat((all_loss_step[phase], loss_step.detach().view(1, -1)))
                            all_loss_experience[phase] = torch.cat((all_loss_experience[phase], loss_experience.detach().view(1, -1)))
                            all_loss_rsd[phase] = torch.cat((all_loss_rsd[phase], loss_rsd.detach().view(1, -1)))
                        if phase in validation_phases:
                            hard_prediction = torch.argmax(step_prediction.detach(), dim=1).cpu().numpy()
                            hard_prediction_exp = torch.argmax(experience_prediction.detach(), dim=1).cpu().numpy()
                            conf_mat[phase] += confusion_matrix(y.cpu().numpy(), hard_prediction, labels=np.arange(n_step_classes))
                            conf_mat_exp[phase] += confusion_matrix(y_experience.cpu().numpy(), hard_prediction_exp, labels=np.arange(2))

                all_loss[phase] = all_loss[phase].cpu().numpy().mean()
                all_loss_step[phase] = all_loss_step[phase].cpu().numpy().mean()
                all_loss_experience[phase] = all_loss_experience[phase].cpu().numpy().mean()
                all_loss_rsd[phase] = all_loss_rsd[phase].cpu().numpy().mean()
                if phase in validation_phases:
                    precision, recall, f1, accuracy = conf2metrics(conf_mat[phase])
                    accuracy_exp[phase] = conf2metrics(conf_mat_exp[phase])[3]
                    all_precision[phase] = precision
                    all_recall[phase] = recall
                    average_precision[phase] = np.mean(all_precision[phase])
                    average_recall[phase] = np.mean(all_recall[phase])
                    all_f1[phase] = f1
                    average_f1[phase] = np.mean(all_f1[phase])

                if log:
                    log_epoch = step_count*epoch+epoch
                    wandb.log({'epoch': log_epoch, f'{phase}/loss': all_loss[phase],
                               f'{phase}/loss_rsd': all_loss_rsd[phase],
                               f'{phase}/loss_step': all_loss_step[phase],
                               f'{phase}/loss_exp': all_loss_experience[phase]})
                    if ((epoch % config['val']['test_every']) == 0) & (phase in validation_phases):
                        wandb.log({'epoch': log_epoch, f'{phase}/precision': average_precision[phase],
                                   f'{phase}/recall': average_recall[phase], f'{phase}/f1': average_f1[phase],
                                   f'{phase}/exp_acc': accuracy_exp[phase]})
            log_text = '%s ([%d/%d] %d%%), train loss: %.4f val loss: %.4f lp: %.4f le: %.4f' % \
                       (timeSince(start_time, (epoch + 1) / stop_epoch),
                        epoch + 1, stop_epoch, (epoch + 1) / stop_epoch * 100,
                        all_loss['train'], all_loss['val'], all_loss_step['val'], all_loss_experience['val'])
            log_text += ' val precision: {0:.4f}, recall: {1:.4f}, f1: {2:.4f}, acc_exp: {3:.4f}'.format(average_precision['val'],
                                                                                       average_recall['val'],
                                                                                       average_f1['val'],
                                                                                        accuracy_exp['val'])
            print(log_text, end='')

            # if current loss is the best we've seen, save model state
            if num_devices > 1:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            if all_loss["val"] < best_loss_on_val:
                # if current loss is the best we've seen, save model state
                non_improving_val_counter = 0
                best_loss_on_val = all_loss["val"]
                print('  **')
                state = {'epoch': epoch + 1,
                         'model_dict': state_dict,
                         'remaining_steps': remaining_steps}

                torch.save(state, output_model_name)
                if log:
                    wandb.summary['best_epoch'] = epoch + 1
                    wandb.summary['best_loss_on_val'] = best_loss_on_val
                    wandb.summary['f1'] = average_f1['val']
                    wandb.summary['exp_acc'] = accuracy_exp['val']

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
    args = parser.parse_args()

    main(output_folder=args.out, log=args.log, pretrained_model=args.pretrained)
