import torch
import os
import glob
import numpy as np
import cv2


class DatasetNoLabel(torch.utils.data.Dataset):
    """
    Dataset for folders with sampled png images from videos
    """
    def __init__(self, datafolders, img_transform=None, max_len=20, fps=2.5):
        super(DatasetNoLabel).__init__()
        self.datafolders = datafolders
        self.img_transform = img_transform
        self.max_len = max_len*fps*60.0
        self.frame2min = 1/(fps*60.0)
        # glob files
        self.surgery_length = {}
        img_files = []
        for d in datafolders:
            files = sorted(glob.glob(os.path.join(d, '*.png')))
            img_files += files
            patientID, frame = self._name2id(files[-1])
            self.surgery_length[patientID] = float(frame)*self.frame2min
        img_files = sorted(img_files)
        assert len(img_files) > 0, 'no png images found in {0}'.format(datafolders)

        self.img_files = img_files
        self.nitems = len(self.img_files)

    def __getitem__(self, index):
        # load image
        ret = cv2.imread(self.img_files[index])
        if ret is None:
            print('weird: ', self.img_files[index])
        img = cv2.cvtColor(cv2.imread(self.img_files[index]), cv2.COLOR_BGR2RGB)

        patientID, frame_number = self._name2id(self.img_files[index])
        elapsed_time = frame_number*self.frame2min
        rsd = self.surgery_length[patientID] - elapsed_time
        if self.img_transform is not None:
            img = self.img_transform(img)
        time_stamp = torch.ones((1, img.shape[1], img.shape[2])) * frame_number / self.max_len
        # append to image as additional channel
        img = torch.cat((img, time_stamp.float()), dim=0)
        return img, elapsed_time, frame_number, rsd

    def __len__(self):
        return self.nitems

    def _name2id(self, filename):
        *patientID, frame = os.path.splitext(os.path.basename(filename))[0].split('_')
        patientID = '_'.join(patientID)
        return patientID, int(frame) - 1


class DatasetCataract101(DatasetNoLabel):
    """
    Dataset for folders with sampled png images from videos
    """
    def __init__(self, datafolders, label_files, img_transform=None, max_len=20, fps=2.5):
        super().__init__(datafolders, img_transform, max_len, fps)
        assert len(label_files) == len(datafolders), 'not the same number of data and label files'
        self.label_files = {}
        for f in label_files:
            patientID = os.path.splitext(os.path.basename(f))[0]
            self.label_files[patientID] = np.genfromtxt(f, delimiter=',', skip_header=1)[:, 1:]

    def __getitem__(self, index):
        img, elapsed_time, frame_number, rsd = super().__getitem__(index)

        # load label
        patientID, frame_number = self._name2id(self.img_files[index])
        label = self.label_files[patientID][frame_number]
        return img, label


class DatasetPartiallyLabeled(DatasetNoLabel):
    """
    Dataset for folders with sampled png images from videos
    """
    def __init__(self, datafolders, label_files, img_transform=None, max_len=20, fps=2.5):
        super().__init__(datafolders, img_transform, max_len, fps)
        self.label_files = {}
        for f in label_files:
            patientID = os.path.splitext(os.path.basename(f))[0]
            self.label_files[patientID] = np.genfromtxt(f, delimiter=',', skip_header=1)[:, 1:]

    def __getitem__(self, index):
        img, elapsed_time, frame_number, rsd = super().__getitem__(index)

        # load label
        patientID, frame_number = self._name2id(self.img_files[index])
        # check if there exists a label for the patient ID
        if patientID in self.label_files:
            label = self.label_files[patientID][frame_number]
            step = label[0]
            experience = label[2] - 1
        else:
            step = -1
            experience = -1
        return img, elapsed_time, frame_number, rsd, step, experience
