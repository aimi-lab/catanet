import numpy as np
from torch import nn
import torch
from unet import UNet


class RoiDetector(nn.Module):
    """ ROI Detector.

           UNET based segmentation model, crops ROI.

           Parameters
           ----------
           checkpoint: dict,
            checkpoint of trained UNET model

          """
    def __init__(self, checkpoint, img_shape=(320,576)):
        super().__init__()
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
        self.unet = UNet(out_classes=checkpoint['n_classes'],
                         in_channels=3,
                         padding=True,
                         padding_mode=checkpoint['model_params']['padding_mode'],
                         num_encoding_blocks=checkpoint['model_params']['depth'],
                         upsampling_type='conv',
                         normalization='batch',
                         out_channels_first_layer=checkpoint['model_params']['n_out_first'])

        self.unet.load_state_dict(checkpoint['model_dict'])
        self.unet.eval()
        self.scaler = nn.InstanceNorm2d(3)  # Normalize image by channel mean and standard deviation.
        self.img_shape = img_shape

    def forward(self, img_in, padding=10, fit_square=False):
        # take numpy array and prepare for segmentation
        img_in = torch.from_numpy(np.moveaxis(img_in, 2, 0)).float()
        img_in = img_in[None, ::]
        # resize to model training size
        orig_shape = img_in.shape[2:]
        X = torch.nn.functional.interpolate(img_in, self.img_shape, mode='area')
        # find foreground mask
        X = self.scaler(X)  # normalize image channel wise
        prediction = self.unet(X)  # forward pass for prediction
        label = torch.argmax(prediction, dim=1)
        # upscale label map back to original image shape
        label = torch.nn.functional.interpolate(label[None, ::].float(), orig_shape,
                                                        mode='nearest').long().squeeze()
        # find edges
        ids = np.where(label > 0)
        if len(ids[0]) > 0:
            corners = np.asarray([[np.min(ids[1])-padding, np.min(ids[0])-padding],
                                  [np.max(ids[1])+padding, np.max(ids[0])+padding]])
        else:
            return None
        if fit_square:
            #  ToDo: constrain to max image shape
            diff = np.max((corners[1][0]-corners[0][0], corners[1][1]-corners[0][1])) - \
                   np.min((corners[1][0]-corners[0][0], corners[1][1]-corners[0][1]))
            id = np.argmax((corners[0][0]-corners[1][0], corners[0][1]-corners[1][1]))
            corners[0, id] -= int(diff/2)
            corners[1, id] += int(diff/2)

        # shift rectangle back into image space
        shift_hor = np.min((0, corners[0, 0], orig_shape[1]-corners[1, 0]))
        shift_ver = np.min((0, corners[0, 1], orig_shape[0]- corners[1, 1]))
        corners[:, 0] -= shift_hor
        corners[:, 1] -= shift_ver
        return corners

# detector = RoiDetector('/home/mhayoz/Cataract_HSS/Code_HSS/hss-cataract-reg/segmentation_models/hssstudy_intraop_unet_model_trained.pth')
# img = cv2.cvtColor(cv2.imread('../data/cataract101/train/269/269_0400.png'), cv2.COLOR_BGR2RGB)
#
# #edges = detector(torch.tensor(img).unsqueeze(0).float().permute(0,3,1,2))
# edges = detector(img, fit_square=True)
# print(edges)
# image = cv2.rectangle(img, tuple(edges[0]), tuple(edges[1]), [0,0,255], 2)
# cv2.imshow('bb', image)
# cv2.waitKey(0)
