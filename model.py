import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_dim, kernel_size, in_channels=9, conv_channels=16, num_fc=1, fc_channels=32, stride=1, padding='same', dilation=1, padding_mode='zeros', output_dim=6, num_preR=1):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.num_fc = num_fc
        self.fc_channels = fc_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.output_dim = output_dim
        self.num_preR = num_preR
        
        self.batch_norm = nn.BatchNorm1d(self.in_channels)

        self.conv1 = nn.Conv1d(self.in_channels, self.conv_channels, self.kernel_size, stride=1, padding=self.padding, dilation=1, padding_mode='zeros')
        self.conv2 = nn.Conv1d(self.conv_channels, self.conv_channels//2, self.kernel_size, stride=1, padding=self.padding, dilation=1, padding_mode='zeros')
        self.conv3 = nn.Conv1d(self.conv_channels//2, self.conv_channels, self.kernel_size, stride=1, padding=self.padding, dilation=1, padding_mode='zeros')

        self.fc = nn.Linear(self.conv_channels*self.input_dim, self.fc_channels)

        if self.num_fc > 1:
            self.linears = nn.ModuleList([nn.Linear(self.fc_channels, self.fc_channels) for i in range(self.num_fc-1)])

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='leaky_relu')
        nn.init.zeros_(self.fc.bias)

        self.out = nn.Linear(self.num_preR*self.fc_channels, output_dim)

    def embed(self, x):

        x = self.batch_norm(x)

        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        x = F.leaky_relu(x)

        x = torch.flatten(x, 1) 
        
        x = self.fc(x)
        x = F.leaky_relu(x)
        
        if self.num_fc > 1:
            for layer in self.linears:
                x = layer(x)
                x = F.leaky_relu(x)

        return x

    def forward(self, x):

        x = self.out(x)
        
        return x