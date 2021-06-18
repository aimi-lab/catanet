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
from models.catRSDNet import CatRSDNet
from utils.dataset_utils import DatasetCataract101
from utils.logging_utils import timeSince
import glob
from sklearn.metrics import confusion_matrix
from torchvision.transforms import Compose, RandomResizedCrop, RandomVerticalFlip, RandomHorizontalFlip, ToPILImage, \
    ToTensor, Resize


def main(output_folder, log, basepath):
    # specify videos of surgeons for training and validation
    config = {'train': {}, 'val': {}, 'data': {}}
    config["train"]['batch_size'] = 50
    config["train"]['epochs'] = 3
    config["train"]['weighted_loss'] = True
    config['train']['sub_epoch_validation'] = 100
    config['train']['learning_rate'] = 0.0001
    config["val"]['batch_size'] = 150
    config['input_size'] = [224, 224]
    config['data']['base_path'] = basepath

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # specify if we should use a GPU (cuda) or only the CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_devices = torch.cuda.device_count()

    # --- training params
    n_step_classes = 11
    training_phases = ['train', 'val']
    img_transform = {}
    img_transform['train'] = Compose([ToPILImage(), RandomHorizontalFlip(), RandomVerticalFlip(),
                                      RandomResizedCrop(size=config['input_size'][0], scale=(0.4,1.0), ratio=(1.0,1.0)),
                                      ToTensor()])
    img_transform['val'] = Compose([ToPILImage(), Resize(config['input_size']), ToTensor()])

    # --- logging
    if log:
        run = wandb.init(project='cataract_rsd', group='catnet')
        run.config.data = config['data']['base_path']
        run.name = run.id
    # --- glob data set
    dataLoader = {}
    for phase in training_phases:
        data_folders = sorted(glob.glob(os.path.join(config['data']['base_path'], phase, '*')))
        labels = sorted(glob.glob(os.path.join(config['data']['base_path'], phase, '**', '*.csv')))
        dataset = DatasetCataract101(data_folders, img_transform=img_transform[phase], label_files=labels)
        dataLoader[phase] = DataLoader(dataset, batch_size=config[phase]['batch_size'],
                                       shuffle=(phase == 'train'), num_workers=4, pin_memory=True)

    output_model_name = os.path.join(output_folder, 'catRSDNet_CNN.pth')

    print('start training... ')
    # --- model
    base_model = CatRSDNet()
    model = base_model.cnn

    if num_devices > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    # --- optimizer
    optim = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    # --- loss
    # loss function
    if config['train']['weighted_loss']:
        label_sum = np.zeros(n_step_classes)
        for fname_label in glob.glob(os.path.join(config['data']['base_path'], 'train', '**', '*.csv')):
            labels = np.genfromtxt(fname_label, delimiter=',', skip_header=1)[:, 1]
            for l in range(n_step_classes):
                label_sum[l] += np.sum(labels==l)
        loss_weights = 1 / label_sum
        loss_weights[label_sum == 0] = 0.0
        loss_weights = torch.tensor(loss_weights / np.max(loss_weights)).float().to(device)
    else:
        loss_weights = None
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    expertise_criterion = nn.CrossEntropyLoss()

    # --- training
    best_loss_on_test = np.Infinity
    start_time = time.time()
    stop_epoch = config['train']['epochs']

    for epoch in range(stop_epoch):
        #zero out epoch based performance variables
        all_loss_train = torch.zeros(0).to(device)
        model.train()  # Set model to training mode

        for ii, (img, labels) in enumerate(dataLoader['train']):
            img = img.to(device)  # input data
            step_label = labels[:, 0].long().to(device)
            expertise = labels[:, 2].long().to(device) - 1
            with torch.set_grad_enabled(True):
                prediction, expertise_pred = model(img)
                loss = criterion(prediction, step_label) + expertise_criterion(expertise_pred, expertise)
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
                    conf_mat = np.zeros((11, 11))
                    conf_mat_exp = np.zeros((2, 2))
                    for jj, (img, label) in enumerate(dataLoader['val']):  # for each of the batches
                        img = img.to(device)  # input data
                        step_label = label[:, 0].long().to(device)
                        expertise = (label[:, 2] - 1).long().to(device)
                        prediction, expertise_pred = model(img)  # [batch size, n_classes]
                        loss = criterion(prediction, step_label) + expertise_criterion(expertise_pred, expertise)
                        val_subepoch_loss = torch.cat((val_subepoch_loss, loss.detach().view(1, -1)))
                        hard_prediction = torch.argmax(prediction.detach(), dim=1).cpu().numpy()
                        conf_mat += confusion_matrix(step_label.cpu().numpy(), hard_prediction,
                                                            labels=np.arange(11))
                        exp_pred = torch.argmax(expertise_pred.detach(), dim=1).cpu().numpy()
                        conf_mat_exp += confusion_matrix(expertise.cpu().numpy(), exp_pred,
                                                            labels=np.arange(2))
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

    parser = argparse.ArgumentParser(description='Training CNN for Cataract Tool Detection')

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
        '--basepath',
        type=str,
        default='data/cataract101',
        help='path to data.'
    )
    args = parser.parse_args()

    main(output_folder=args.out, log=args.log, basepath=args.basepath)
