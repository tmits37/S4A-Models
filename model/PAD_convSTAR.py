# Implementation of the convSTAR model.
# Code taken from:
#   https://github.com/0zgur0/ms-convSTAR

from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.nn import init
from tensorboardX import SummaryWriter
import pytorch_lightning as pl

from .encoder_decoder import EncoderDecoder

class ConvSTARCell(nn.Module):
    """
    Generate a convolutional STAR cell
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal(self.update.weight)
        init.orthogonal(self.gate.weight)
        init.constant(self.update.bias, 0.)
        init.constant(self.gate.bias, 1.)

        print('convSTAR cell is constructed with h_dim: ', hidden_size)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = torch.zeros(state_size).cuda()
            else:
                prev_state = torch.zeros(state_size)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        gain = torch.sigmoid( self.gate(stacked_inputs) )
        update = torch.tanh( self.update(input_) )
        new_state = gain * prev_state + (1-gain) * update

        return new_state


class ConvSTAR(EncoderDecoder):

    def __init__(self, run_path, linear_encoder, learning_rate=1e-3, parcel_loss=False,
                 class_weights=None, crop_encoding=None, checkpoint_epoch=None,
                 input_size=4, n_layers=3, kernel_sizes=3, hidden_sizes=64):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.

        Parameters
        ----------
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
        input_size: int, default 4
            Depth dimension of input tensors.
        hidden_sizes: int or list, default 64
            Depth dimensions of hidden state. If integer, the same hidden size is used for all cells.
        kernel_sizes: int or list, default 3
            Sizes of Conv2d gate kernels. If integer, the same kernel size is used for all cells.
        n_layers: int, default 3
            Number of chained `ConvSTARCell`.
        '''

        super(ConvSTAR, self).__init__(
                 run_path=run_path,
                 linear_encoder=linear_encoder, 
                 learning_rate=learning_rate, 
                 parcel_loss=parcel_loss,
                 class_weights=class_weights, 
                 crop_encoding=crop_encoding, 
                 checkpoint_epoch=checkpoint_epoch)


        self.input_size = input_size
        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes]*n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes]*n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]

            cell = ConvSTARCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
            name = 'ConvSTARCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

        self.conv2d = nn.Conv2d(in_channels=self.hidden_sizes[-1],
                                out_channels=self.num_discrete_labels,
                                kernel_size=1,
                                stride=1,
                                padding=0)

        self.softmax = nn.LogSoftmax(dim=1)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.save_hyperparameters()


    def forward(self, x, hidden=None):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).

        Returns
        -------
        The prediction of the last hidden layer.
        '''
        if not hidden:
            hidden = [None] * self.n_layers

        b, d, h, w = x.shape
        x = x.view(b, -1, self.input_size, h, w)
        x = x.transpose(1, 2)   # (B, T, C, H, W) -> (B, C, T, H, W)

        batch_size, channels, timesteps, height, width = x.size()

        # retain tensors in list to allow different hidden sizes
        upd_hidden = []

        for timestep in range(timesteps):
            input_ = x[:, :, timestep, :, :]

            for layer_idx in range(self.n_layers):
                cell = self.cells[layer_idx]
                cell_hidden = hidden[layer_idx]

                if layer_idx == 0:
                    upd_cell_hidden = cell(input_, cell_hidden)
                else:
                    upd_cell_hidden = cell(upd_hidden[-1], cell_hidden)
                upd_hidden.append(upd_cell_hidden)
                # update input_ to the last updated hidden layer for next pass
                hidden[layer_idx] = upd_cell_hidden

        # Keep only the last output for an N-to-1 scheme
        x = hidden[-1]   # (L, B, C, H, W) -> (B, C, H, W)
        x = self.softmax(self.conv2d(x))   # (B, K, H, W)

        return x


class ConvSTAR_Res(nn.Module):

    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.

        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvSTARCell`.
        '''

        super(ConvSTAR_Res, self).__init__()

        self.input_size = input_size

        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes] * n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes] * n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            elif i == 2 or i==4:
                input_dim = self.hidden_sizes[i - 1] + self.input_size
            else:
                input_dim = self.hidden_sizes[i - 1]

            cell = ConvSTARCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
            name = 'ConvSTARCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

    def forward(self, x, hidden=None):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).
        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''
        if not hidden:
            hidden = [None] * self.n_layers

        input_ = x

        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            if layer_idx==1 or layer_idx==3:
                input_ = torch.cat((upd_cell_hidden, x), 1)
            else:
                input_ = upd_cell_hidden


        # retain tensors in list to allow different hidden sizes
        return upd_hidden
