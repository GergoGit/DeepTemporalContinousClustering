# -*- coding: utf-8 -*-
"""
Created on Mon May 23 08:27:36 2022

@author: bonnyaigergo
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
# import torchinfo
import utils

class EncoderConv1D(nn.Module):
    def __init__(self, 
                 n_timesteps: int, 
                 n_features: int, 
                 embedding_size: int, 
                 batch_size: int,
                 enc_arch_params: dict):
        super().__init__()
        self.n_timesteps = n_timesteps 
        self.n_features = n_features
        self.embedding_size = embedding_size 
        self.batch_size = batch_size
        self.conv_block_output_size = utils.calculate_conv_block_output_size(input_size=n_timesteps, enc_arch_params)
        self.multi_block = enc_arch_params['enc_multi_block']
        self.enc_arch_params = enc_arch_params
        
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=enc_arch_params['conv1_ch_in'], 
                                            out_channels=enc_arch_params['conv1_ch_out'], 
                                            kernel_size=enc_arch_params['conv1_kernel_size'], 
                                            stride=enc_arch_params['conv1_stride'], 
                                            padding=enc_arch_params['conv1_padding']),
                                   nn.BatchNorm1d(enc_arch_params['conv1_ch_out']),
                                   nn.Dropout(0.2),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=enc_arch_params['pool1_kernel_size'], 
                                                stride=enc_arch_params['pool1_stride'], 
                                                padding=enc_arch_params['pool1_padding'])
                                   )
        if self.multi_block:
            self.conv2 = nn.Sequential(nn.Conv1d(in_channels=enc_arch_params['conv1_ch_out'], 
                                                out_channels=enc_arch_params['conv2_ch_out'], 
                                                kernel_size=enc_arch_params['conv2_kernel_size'], 
                                                stride=enc_arch_params['conv2_stride'], 
                                                padding=enc_arch_params['conv2_padding']),
                                       nn.Dropout(0.2),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(enc_arch_params['conv2_ch_out']),
                                       nn.MaxPool1d(kernel_size=enc_arch_params['pool2_kernel_size'], 
                                                    stride=enc_arch_params['pool2_stride'], 
                                                    padding=enc_arch_params['pool2_padding'])
                                       )
        self.flatten = nn.Flatten()     
        self.fc = nn.Linear(self.conv_block_output_size, self.embedding_size)

    def forward(self, x):
        x = x.reshape((self.batch_size, self.n_features, self.n_timesteps))
        x = self.conv1(x)
        if self.multi_block:
            x = self.conv2(x)
        x = self.flatten(x)
        return self.fc(x)
    

class DecoderConv1D(nn.Module):
    def __init__(self, 
                 n_timesteps, 
                 n_features,
                 embedding_size, 
                 batch_size, 
                 dec_arch_params):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.embedding_size = embedding_size
        self.n_features = n_features
        self.batch_size = batch_size
        self.convtr1d_block_output_size = utils.calculate_convtranspose1d_block_output_size(input_size=embedding_size, dec_arch_params)
        self.multi_block = dec_arch_params['dec_multi_block']
        self.dec_arch_params = dec_arch_params
        
        # self.fc1 = nn.Linear(embedding_size, self.embedding_size)        
        self.tr_conv1 = nn.ConvTranspose1d(in_channels=dec_arch_params['convtr1_ch_in'], 
                                           out_channels=dec_arch_params['convtr1_ch_out'], 
                                           kernel_size=dec_arch_params['convtr1_kernel_size'],
                                           stride=dec_arch_params['convtr1_stride'], 
                                           padding=dec_arch_params['convtr1_padding'],
                                           output_padding=dec_arch_params['convtr1_output_padding'])
        if self.multi_block:
            self.tr_conv2 = nn.ConvTranspose1d(in_channels=dec_arch_params['convtr1_ch_out'], 
                                               out_channels=dec_arch_params['convtr2_ch_out'], 
                                               kernel_size=dec_arch_params['convtr2_kernel_size'],
                                               stride=dec_arch_params['convtr2_stride'], 
                                               padding=dec_arch_params['convtr2_padding'],
                                               output_padding=dec_arch_params['convtr2_output_padding'])
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(self.convtr1d_block_output_size, self.n_timesteps)

    def forward(self, x):
        x = x.reshape((self.batch_size, self.n_features, self.embedding_size))
        x = self.tr_conv1(x)
        if self.multi_block:
            x = self.tr_conv2(x)
        x = self.flatten(x)
        return self.output_layer(x)
    
    
class Autoencoder(nn.Module):
    def __init__(self, 
                 encoder: nn.Module, 
                 decoder: nn.Module):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        
    def encoding(self, x):
        return self.encoder(x)


class VariationalAutoEncoder(nn.Module):
    def __init__(self,
                 encoder: nn.Module, 
                 decoder: nn.Module):
        super(VariationalAutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.embedding_size = encoder.embedding_size
        self.z_mean = torch.nn.Linear(self.embedding_size, self.embedding_size)
        self.z_logvar = torch.nn.Linear(self.embedding_size, self.embedding_size)
        
        nn.init.xavier_uniform_(self.z_mean.weight)
        nn.init.xavier_uniform_(self.z_logvar.weight)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        x = self.encoder(x)
        z_mu, z_logvar = self.z_mean(x), self.z_logvar(x)
        encoded = self.reparameterize(z_mu, z_logvar)
        decoded = self.decoder(encoded)
        return z_mu, z_logvar, encoded, decoded
    
    def sampling(self, n_sample):
        z = torch.randn(n_sample, self.embedding_size)
        samples = self.decode(z)
        return samples
        
    def encoding(self, x):
        x = self.encoder(x)
        z_mean, z_logvar = self.z_mean(x), self.z_logvar(x)
        encoded = self.reparameterize(z_mean, z_logvar)
        return encoded
    
    
    
class DilatedCasualConv1D(nn.Module):
    
    def __init__(self, 
                 n_layers, 
                 kernel_size, 
                 dropout_rate, 
                 out_channels):
        super(DilatedCasualConv1D, self).__init__()
        
        dilations = [2**i for i in range(n_layers+1)]
        self.tcn_block = nn.Sequential()
        # TODO: add more channels then downsample
        for i, d in enumerate(dilations):
            self.tcn_block.add_module('conv'+str(i), weight_norm(nn.Conv1d(in_channels=1 if i==0 else out_channels,
                                                                out_channels=out_channels, 
                                                                kernel_size=kernel_size,
                                                                dilation=d)))
            self.tcn_block.add_module('relu'+str(i), nn.ReLU())
            # self.tcn_block.add_module('dropout'+str(i), nn.Dropout(dropout_rate))
    
    def forward(self, x):
        return self.tcn_block(x)
    
class EncoderTCN1D(nn.Module):
    def __init__(self,
                 n_timesteps: int, 
                 n_features: int, 
                 embedding_size: int, 
                 batch_size: int,
                 enc_arch_params: dict):
        super(EncoderTCN1D, self).__init__()
        self.n_timesteps = n_timesteps 
        self.n_features = n_features
        self.embedding_size = embedding_size 
        self.batch_size=batch_size
        self.tcn_block_output_size = utils.calculate_tcn_block_output_size(input_size=n_timesteps, enc_arch_params)
        self.enc_arch_params = enc_arch_params
        
        assert self.tcn_block_output_size > 0, 'TCN block output size has to be positive integer'
        
        self.tcn_block1 = DilatedCasualConv1D(out_channels=enc_arch_params['tcn1_ch_out'],
                                              n_layers=enc_arch_params['tcn1_n_layer'], 
                                              kernel_size=enc_arch_params['tcn1_kernel_size'], 
                                              dropout_rate=enc_arch_params['tcn1_dropout_rate'])
        # self.bn1 = nn.BatchNorm1d(enc_arch_params['conv1_ch_out'])
        # self.drop1 = nn.Dropout(0.5)
        if enc_arch_params['enc_maxpool']:
            self.pool1 = nn.MaxPool1d(kernel_size=enc_arch_params['pool1_kernel_size'], 
                                      stride=enc_arch_params['pool1_stride'], 
                                      padding=enc_arch_params['pool1_padding'])
        else:
            self.pool1 = nn.AvgPool1d(kernel_size=enc_arch_params['pool1_kernel_size'], 
                                      stride=enc_arch_params['pool1_stride'], 
                                      padding=enc_arch_params['pool1_padding'])
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()        
        self.fc = nn.Linear(self.tcn_block_output_size, embedding_size)

    def forward(self, x):
        x = x.reshape((self.batch_size, self.n_features, self.n_timesteps))
        x = self.tcn_block1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        return self.fc(x)
    
class EncoderRNN(nn.Module):
    def __init__(self, 
                 n_timesteps: int, 
                 n_features: int, 
                 embedding_size: int, 
                 batch_size: int,
                 enc_arch_params: dict):
        super(EncoderRNN, self).__init__()
        self.n_timesteps = n_timesteps 
        self.n_features = n_features
        self.embedding_size = embedding_size 
        self.hidden_size = enc_arch_params['hidden_size']
        self.batch_size = batch_size
        self.cell_type = enc_arch_params['cell_type']
        self.multi_block = enc_arch_params['enc_multi_block']
        self.layer1_bidirectional = enc_arch_params['layer1_bidirectional']
        self.layer2_bidirectional = enc_arch_params['layer2_bidirectional']
        self.enc_arch_params = enc_arch_params
        
        if self.cell_type == "LSTM":
            self.lstm1 = nn.LSTM(
                input_size=n_features,
                hidden_size=self.hidden_size,
                num_layers=enc_arch_params['layer1_n_layers'],
                batch_first=True,
                bidirectional=enc_arch_params['layer1_bidirectional'],
                dropout=enc_arch_params['layer1_dropout_rate']
            )
            if self.multi_block:
                self.lstm2 = nn.LSTM(
                    input_size=self.hidden_size * (2 if self.layer1_bidirectional else 1),
                    hidden_size=embedding_size,
                    num_layers=enc_arch_params['layer2_n_layers'],
                    batch_first=True,
                    bidirectional=enc_arch_params['layer2_bidirectional'],
                    dropout=enc_arch_params['layer2_dropout_rate']
                )
        elif self.cell_type == "GRU":
            self.gru1 = nn.GRU(
                input_size=n_features,
                hidden_size=self.hidden_size,
                num_layers=enc_arch_params['layer1_n_layers'],
                batch_first=True,
                bidirectional=enc_arch_params['layer1_bidirectional'],
                dropout=enc_arch_params['layer1_dropout_rate']
            )
            if self.multi_block:
                self.gru2 = nn.GRU(
                    input_size=self.hidden_size * (2 if self.layer1_bidirectional else 1),
                    hidden_size=embedding_size,
                    num_layers=enc_arch_params['layer2_n_layers'],
                    batch_first=True,
                    bidirectional=enc_arch_params['layer2_bidirectional'],
                    dropout=enc_arch_params['layer2_dropout_rate']
                )            
        else:
            raise NotImplementedError    
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=enc_arch_params['before_fc_dropout'])
        if self.multi_block:
            self.fc = nn.Linear(self.embedding_size * (2 if self.layer2_bidirectional else 1), 
                                self.embedding_size)
        else:
            self.fc = nn.Linear(self.hidden_size * (2 if self.layer1_bidirectional else 1), 
                                self.embedding_size)

    def forward(self, x):
        x = x.reshape((self.batch_size, self.n_timesteps, self.n_features))
        if self.cell_type == "LSTM":
            x, (hidden, cell_1) = self.lstm1(x)
        elif self.cell_type == "GRU":
            x, hidden = self.gru1(x)
            
        if self.multi_block:
            if self.cell_type == "LSTM":
                x, (hidden, _) = self.lstm2(x)
            elif self.cell_type == "GRU":
                x, hidden = self.gru2(x)     
                
        if (self.layer1_bidirectional and self.multi_block==False) or self.multi_block and self.layer1_bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
        hidden = self.relu(self.dropout(hidden))
        return self.fc(hidden)
    
    
class DecoderRNN(nn.Module):
    def __init__(self, 
                 n_timesteps: int, 
                 n_features: int, 
                 embedding_size: int, 
                 batch_size: int,
                 dec_arch_params: dict):
        super(DecoderRNN, self).__init__()
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.layer1_output_size = dec_arch_params['layer1_output_size']
        self.layer2_output_size = dec_arch_params['layer2_output_size']
        self.cell_type = dec_arch_params['cell_type']
        self.multi_block = dec_arch_params['dec_multi_block']
        self.layer1_bidirectional = dec_arch_params['layer1_bidirectional']
        self.layer2_bidirectional = dec_arch_params['layer2_bidirectional']
        self.dec_arch_params = dec_arch_params
        
        if self.cell_type == "LSTM":
            self.lstm1 = nn.LSTM(
                input_size=embedding_size,
                hidden_size=self.layer1_output_size,
                num_layers=dec_arch_params['layer1_n_layers'],
                batch_first=True,
                bidirectional=dec_arch_params['layer1_bidirectional'],
                dropout=dec_arch_params['layer1_dropout_rate']
            )
            if self.multi_block:
                self.lstm2 = nn.LSTM(
                    input_size=self.layer1_output_size * (2 if self.layer1_bidirectional else 1),
                    hidden_size=self.layer2_output_size,
                    num_layers=dec_arch_params['layer2_n_layers'],
                    batch_first=True,
                    bidirectional=dec_arch_params['layer2_bidirectional'],
                    dropout=dec_arch_params['layer2_dropout_rate']
                )
        elif self.cell_type == "GRU":
            self.gru1 = nn.GRU(
                input_size=embedding_size,
                hidden_size=self.layer1_output_size,
                num_layers=dec_arch_params['layer1_n_layers'],
                batch_first=True,
                bidirectional=dec_arch_params['layer1_bidirectional'],
                dropout=dec_arch_params['layer1_dropout_rate']
            )
            if self.multi_block:
                self.gru2 = nn.GRU(
                    input_size=self.layer1_output_size * (2 if self.layer1_bidirectional else 1),
                    hidden_size=self.layer2_output_size,
                    num_layers=dec_arch_params['layer2_n_layers'],
                    batch_first=True,
                    bidirectional=dec_arch_params['layer2_bidirectional'],
                    dropout=dec_arch_params['layer2_dropout_rate']
                )
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dec_arch_params['before_fc_dropout'])
        if self.multi_block:
            self.fc = nn.Linear(self.layer2_output_size * (2 if self.layer2_bidirectional else 1), 
                                n_features)
        else:
            self.fc = nn.Linear(self.layer1_output_size * (2 if self.layer1_bidirectional else 1), 
                                n_features)

    def forward(self, x):
        x = x.repeat(self.n_timesteps, self.n_features) 
        x = x.reshape((self.batch_size, self.n_timesteps, self.embedding_size))
        if self.cell_type == "LSTM":
            x, (_, _) = self.lstm1(x)
        elif self.cell_type == "GRU":
            x, _ = self.gru1(x)
        if self.multi_block:
            if self.cell_type == "LSTM":
                x, (_, _) = self.lstm2(x)
            elif self.cell_type == "GRU":
                x, _ = self.gru2(x)
        x = self.relu(self.dropout(x))
        x = self.fc(x)
        return x.reshape((self.batch_size, self.n_timesteps))


class Discriminator(nn.Module):
    def __init__(self,
                 embedding_size: int,
                 discr_arch_params: dict):
        super(Discriminator, self).__init__()
        self.discr_arch_params = discr_arch_params
        self.model = nn.Sequential(
            nn.Linear(embedding_size, discr_arch_params['layer1_out']),
            nn.ReLU(),
            nn.Dropout(discr_arch_params['dropout']),
            nn.Linear(discr_arch_params['layer1_out'], discr_arch_params['layer2_out']),
            nn.ReLU(),
            nn.Dropout(discr_arch_params['dropout']),
            nn.Linear(discr_arch_params['layer2_out'], discr_arch_params['layer3_out']),
            nn.ReLU(),
            nn.Dropout(discr_arch_params['dropout']),
            nn.Linear(discr_arch_params['layer3_out'], 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, 
                 n_timesteps, 
                 n_features,
                 batch_size, 
                 gen_arch_params):
        super(Generator, self).__init__()
        self.n_timesteps = n_timesteps
        self.random_input_size = gen_arch_params['random_input_size']
        self.n_features = n_features
        self.batch_size = batch_size
        self.convtr1d_block_output_size = utils.calculate_convtranspose1d_block_output_size(input_size=self.random_input_size, 
                                                                                            **gen_arch_params)
        self.multi_block = gen_arch_params['dec_multi_block']
        self.gen_arch_params = gen_arch_params
        
        # self.fc1 = nn.Linear(input_size, self.input_size)        
        self.tr_conv1 = nn.ConvTranspose1d(in_channels=gen_arch_params['convtr1_ch_in'], 
                                           out_channels=gen_arch_params['convtr1_ch_out'], 
                                           kernel_size=gen_arch_params['convtr1_kernel_size'],
                                           stride=gen_arch_params['convtr1_stride'], 
                                           padding=gen_arch_params['convtr1_padding'],
                                           output_padding=gen_arch_params['convtr1_output_padding'])
        if self.multi_block:
            self.tr_conv2 = nn.ConvTranspose1d(in_channels=gen_arch_params['convtr1_ch_out'], 
                                               out_channels=gen_arch_params['convtr2_ch_out'], 
                                               kernel_size=gen_arch_params['convtr2_kernel_size'],
                                               stride=gen_arch_params['convtr2_stride'], 
                                               padding=gen_arch_params['convtr2_padding'],
                                               output_padding=gen_arch_params['convtr2_output_padding'])
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(self.convtr1d_block_output_size, self.n_timesteps)

    def forward(self, x):
        x = x.reshape((self.batch_size, self.n_features, self.random_input_size))
        x = self.tr_conv1(x)
        if self.multi_block:
            x = self.tr_conv2(x)
        x = self.flatten(x)
        return self.output_layer(x)