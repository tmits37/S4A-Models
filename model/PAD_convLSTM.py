'''
Implementation of the model proposed in:
- Xingjian Shi, et al. "Convolutional LSTM Network: A Machine Learning
Approach for Precipitation Nowcasting." (2015), arXiv:1506.04214v2.

Code adopted from:
https://github.com/jhhuang96/ConvLSTM-PyTorch.git
'''

import os

import numpy as np
from tqdm import tqdm
import copy
from pathlib import Path
import pickle

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
from tensorboardX import SummaryWriter
import pytorch_lightning as pl

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

from .encoder_decoder import EncoderDecoder


def print_model_stats(model):
    '''
    Prints the number of parameters, the number of trainable parameters and
    the peak memory usage of a given Pytorch model.
    '''
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {pytorch_total_params}')
    print(f'Trainable params: {pytorch_train_params}')
    print(f'Peak memory usage: {(torch.cuda.max_memory_allocated() / (1024**2)):.2f} MB\n')


def tensor_size(t):
    size = t.element_size() * t.nelement()
    print(f'\nTensor memory usage: {size / (1024**2):.2f} MB')


def get_last_model_checkpoint(path):
    '''
    Browses through the given path and finds the last saved checkpoint of a
    model.

    Parameters
    ----------
    path: str or Path
        The path to search.

    Returns
    -------
    (Path, Path, int): the path of the last model checkpoint file, the path of the
    last optimizer checkpoint file and the corresponding epoch.
    '''
    model_chkp = [c for c in Path(path).glob('model_state_dict_*')]
    optimizer_chkp = [c for c in Path(path).glob('optimizer_state_dict_*')]
    model_chkp_per_epoch = {int(c.name.split('.')[0].split('_')[-1]): c for c in model_chkp}
    optimizer_chkp_per_epoch = {int(c.name.split('.')[0].split('_')[-1]): c for c in optimizer_chkp}

    last_model_epoch = sorted(model_chkp_per_epoch.keys())[-1]
    last_optimizer_epoch = sorted(optimizer_chkp_per_epoch.keys())[-1]

    assert last_model_epoch == last_optimizer_epoch, 'Error: Could not resume training. Optimizer or model checkpoint missing.'

    return model_chkp_per_epoch[last_model_epoch], optimizer_chkp_per_epoch[last_model_epoch], last_model_epoch


class CLSTM_cell(nn.Module):
    """
    The ConvLSTM cell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # height, width
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      4 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features))


    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        if hidden_state is None:
            hx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).cuda()
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).cuda()
        else:
            hx, cx = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.input_channels, self.shape[0],
                                self.shape[1]).cuda()
            else:
                x = inputs[index, ...]

            combined = torch.cat((x, hx), 1)
            gates = self.conv(combined)
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        return torch.stack(output_inner), (hy, cy)


class ConvLSTM(EncoderDecoder):
    def __init__(self, run_path, linear_encoder, learning_rate=1e-3, parcel_loss=False,
                 class_weights=None, crop_encoding=None, checkpoint_epoch=None):
        '''
        Parameters:
        -----------
        run_path: str or Path
            The path to export results into.
        linear_encoder: dict
            A dictionary mapping the true labels to the given labels.
            True labels = the labels in the mappings file.
            Given labels = labels ranging from 0 to len(true labels), which the
            true labels have been converted into.
        learning_rate: float, default 1e-3
            The initial learning rate.
        parcel_loss: boolean, default False
            If True, then a custom loss function is used which takes into account
            only the pixels of the parcels. If False, then all image pixels are
            used in the loss function.
        class_weights: dict, default None
            Weights per class to use in the loss function.
        crop_encoding: dict, default None
            A dictionary mapping class ids to class names.
        checkpoint_epoch: int, default None
            The epoch loaded for testing.
        '''
        super(ConvLSTM, self).__init__(
                 run_path=run_path,
                 linear_encoder=linear_encoder, 
                 learning_rate=learning_rate, 
                 parcel_loss=parcel_loss,
                 class_weights=class_weights, 
                 crop_encoding=crop_encoding, 
                 checkpoint_epoch=checkpoint_epoch)



        self.input_size = 4
        # Encoder
        # -------
        self.conv_1e = nn.Conv2d(in_channels=self.input_size,
                                out_channels=16,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.leakyrelu_1e = nn.LeakyReLU(negative_slope=0.2)
        self.clstm_1e = CLSTM_cell(shape=(64, 64), input_channels=16, filter_size=5, num_features=64)

        self.conv_2e = nn.Conv2d(in_channels=64,
                                out_channels=64,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        self.leakyrelu_2e = nn.LeakyReLU(negative_slope=0.2)
        self.clstm_2e = CLSTM_cell(shape=(32, 32), input_channels=64, filter_size=5, num_features=96)

        self.conv_3e = nn.Conv2d(in_channels=96,
                                out_channels=96,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        self.leakyrelu_3e = nn.LeakyReLU(negative_slope=0.2)
        self.clstm_3e = CLSTM_cell(shape=(16, 16), input_channels=96, filter_size=5, num_features=96)

        # Decoder
        # -------
        self.clstm_1d = CLSTM_cell(shape=(16, 16), input_channels=96, filter_size=5, num_features=96)
        self.transconv_1d = nn.ConvTranspose2d(in_channels=96,
                                             out_channels=96,
                                             kernel_size=4, # 3
                                             stride=2,
                                             padding=3)  # 2
        self.leakyrelu_1d = nn.LeakyReLU(negative_slope=0.2)

        self.clstm_2d = CLSTM_cell(shape=(32, 32), input_channels=96, filter_size=5, num_features=96)
        self.transconv_2d = nn.ConvTranspose2d(in_channels=96,
                                             out_channels=96,
                                             kernel_size=4,
                                             stride=2,
                                             padding=3)
        self.leakyrelu_2d = nn.LeakyReLU(negative_slope=0.2)

        self.clstm_3d = CLSTM_cell(shape=(64, 64), input_channels=96, filter_size=5, num_features=64)
        self.transconv_3d = nn.ConvTranspose2d(in_channels=64,
                                             out_channels=16,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1)
        self.leakyrelu_3d = nn.LeakyReLU(negative_slope=0.2)

        self.transconv_4d = nn.ConvTranspose2d(in_channels=16,
                                             out_channels=self.num_discrete_labels,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0)
        self.softmax = nn.LogSoftmax(dim=1)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.save_hyperparameters()


    def forward(self, input):
        
        b, d, h, w = input.shape
        input = input.view(b, -1, self.input_size, h, w)
        input = input.transpose(0, 1)

        hidden_states = []

        # Encoder
        num_timesteps, batch_size, input_channels, height, width = input.size()
        input = torch.reshape(input, (-1, input_channels, height, width))
        x = self.leakyrelu_1e(self.conv_1e(input))
        x = torch.reshape(x, (num_timesteps, batch_size, x.size(1), x.size(2), x.size(3)))
        x, hidden_state = self.clstm_1e(x, None, num_timesteps)
        hidden_states.append(hidden_state)

        num_timesteps, batch_size, input_channels, height, width = x.size()
        x = torch.reshape(x, (-1, input_channels, height, width))
        x = self.leakyrelu_2e(self.conv_2e(x))
        x = torch.reshape(x, (num_timesteps, batch_size, x.size(1), x.size(2), x.size(3)))
        x, hidden_state = self.clstm_2e(x, None, num_timesteps)
        hidden_states.append(hidden_state)

        num_timesteps, batch_size, input_channels, height, width = x.size()
        x = torch.reshape(x, (-1, input_channels, height, width))
        x = self.leakyrelu_3e(self.conv_3e(x))
        x = torch.reshape(x, (num_timesteps, batch_size, x.size(1), x.size(2), x.size(3)))
        x, hidden_state = self.clstm_3e(x, None, num_timesteps)
        hidden_states.append(hidden_state)

        # Decoder
        x, hidden_state = self.clstm_1d(None, hidden_states[-1], x.size(0))
        num_timesteps, batch_size, input_channels, height, width = x.size()
        x = torch.reshape(x, (-1, input_channels, height, width))
        x = self.leakyrelu_1d(self.transconv_1d(x))
        x = torch.reshape(x, (num_timesteps, batch_size, x.size(1), x.size(2), x.size(3)))

        x, hidden_state = self.clstm_2d(None, hidden_states[-2], x.size(0))
        num_timesteps, batch_size, input_channels, height, width = x.size()
        x = torch.reshape(x, (-1, input_channels, height, width))
        x = self.leakyrelu_2d(self.transconv_2d(x))
        x = torch.reshape(x, (num_timesteps, batch_size, x.size(1), x.size(2), x.size(3)))

        x, hidden_state = self.clstm_3d(None, hidden_states[-3], x.size(0))
        num_timesteps, batch_size, input_channels, height, width = x.size()
        x = torch.reshape(x, (-1, input_channels, height, width))
        x = self.leakyrelu_3d(self.transconv_3d(x))

        x = self.transconv_4d(x)
        x = torch.reshape(x, (num_timesteps, batch_size, x.size(1), x.size(2), x.size(3)))

        # Keep only the last output for an N-to-1 scheme
        x = x[-1, :, :, :, :]  # (T, B, K, H, W) -> (B, K, H, W)
        x = self.softmax(x)

        return x


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        pla_lr_scheduler = {
            'scheduler': lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        factor=0.5,
                                                        patience=4,
                                                        verbose=True),
            'monitor': 'val_loss'
        }
        return [optimizer], [pla_lr_scheduler]
