'''
Implementation of the model proposed in:
- Charlotte Pelletier, et al. "Temporal Convolutional Neural Network for the
Classification of Satellite Image Time Series." (2018), arXiv:1811.10166.

Code adopted from:
https://github.com/MarcCoru/crop-type-mapping
'''

import torch
import torch.nn as nn
import torch.utils.data

from .encoder_decoder import EncoderDecoder

class Conv1D_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=5, drop_probability=0.5):
        super(Conv1D_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims, kernel_size, padding=(kernel_size // 2)),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class FC_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, drop_probability=0.5):
        super(FC_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class TempCNN(EncoderDecoder):
    def __init__(self, input_dim, nclasses, sequence_length, run_path, linear_encoder,
                 learning_rate=0.001, kernel_size=5, hidden_dims=64, dropout=0.5,
                 weight_decay=1e-6, parcel_loss=False, class_weights=None):

        super(TempCNN, self).__init__(
                 run_path=run_path,
                 linear_encoder=linear_encoder, 
                 learning_rate=learning_rate, 
                 parcel_loss=parcel_loss,
                 class_weights=class_weights, 
                 crop_encoding=None, 
                 checkpoint_epoch=None)


        if not parcel_loss:
            if class_weights is None:
                self.lossfunction = nn.NLLLoss()
            else:
                self.lossfunction = nn.NLLLoss(weight=torch.as_tensor(class_weights.values()))
        else:
            if class_weights is None:
                self.lossfunction = nn.NLLLoss(reduction='sum')
            else:
                self.lossfunction = nn.NLLLoss(weight=torch.as_tensor(list(class_weights.values())), reduction='sum')


        self.hidden_dims = hidden_dims
        self.sequence_length = sequence_length

        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(input_dim, hidden_dims, kernel_size=kernel_size, drop_probability=dropout)
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size, drop_probability=dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size, drop_probability=dropout)
        self.flatten = Flatten()
        self.dense = FC_BatchNorm_Relu_Dropout(hidden_dims*sequence_length, 4*hidden_dims, drop_probability=dropout)
        self.logsoftmax = nn.Sequential(nn.Linear(4 * hidden_dims, nclasses), nn.LogSoftmax(dim=-1))

        # Hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()


    def forward(self,x):
        b, t, c, h, w = x.size()
        x = x.view(b, -1, h, w)

        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.logsoftmax(x)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=self.weight_decay, lr=self.learning_rate)
        return [optimizer]