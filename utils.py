# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 01:22:19 2022

@author: bonnyaigergo

EarlyStopping:
https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
from math import floor

def plot_loss(data: dict, train='train', val="val", log_yscale: bool=True):
    plt.plot(data[train])
    if val != None:
        plt.plot(data[val])
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.title('Loss over training epochs')
    if val != None:
        plt.legend(['train','val'])
    else:
        plt.legend([train])
    if log_yscale:
        plt.yscale('log')
    plt.show()
    
def embedding_histogram(model: nn.Module, 
                        dataloader: DataLoader, 
                        n_batch: int, 
                        n_rows: int, 
                        n_cols: int):
    batch_counter = 0
    embeddings = []
    for i, seq in enumerate(dataloader):
        with torch.no_grad():
            embedding = model.encoding(seq).to('cpu')
        
        embeddings.append(embedding)
        batch_counter += 1
        if batch_counter >= n_batch:
            break
    
    result = torch.cat(embeddings, dim=0)
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, 
                             sharex=True, sharey=True,
                             # figsize=(15, 15)
                             )
    
    i = 0
    for row in range(len(axes)):
        for col in range(len(axes[0])):
            axes[row][col].hist(result[:, i].numpy())
            i += 1
    plt.show()

def plot_generated_vs_original(dataloader: DataLoader, 
                               model: nn.Module, 
                               seq_number: int):
    """
    Parameters
    ----------
    dataloader : DataLoader
    model : nn.Module
        Defined network.
    seq_number : int
        integer less then or equal to batch_size.

    Returns
    -------
    Line chart of the original sample and the generated one.
    """
    seq_true = next(iter(dataloader))
    seq_pred = model.forward(seq_true)
    
    plt.plot(seq_true[seq_number].detach().numpy())
    plt.plot(seq_pred[seq_number].detach().numpy())
    plt.ylabel('y')
    plt.xlabel('time')
    plt.title('Sample check')
    plt.legend(['real','generated'])
    plt.show()
    
class EarlyStopping():
    """
    Early stopping to terminate the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience: int=5, delta: float=0, verbose: bool=False):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.best_model = None
        self.meet_criterion = False
        
    def __call__(self, val_loss_epoch, model):
        if self.best_loss == None:
            self.best_loss = val_loss_epoch
        elif self.best_loss - val_loss_epoch > self.delta:
            self.best_loss = val_loss_epoch
            self.best_model_weights = deepcopy(model.state_dict())
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss_epoch < self.delta:
            self.counter += 1
            if self.verbose:
                print(f"Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    print('Early stopping')
                self.meet_criterion = True
                
                
def calc_conv_or_pool_output_size(input_size: int, kernel_size: int, stride: int, padding: int):
    output_size = floor((input_size + 2*padding - kernel_size)/stride) + 1
    return output_size

def calculate_conv_block_output_size(input_size: int, enc_arch_params: dict):
    
    conv1_output_size = calc_conv_or_pool_output_size(input_size=input_size, 
                                                        kernel_size=enc_arch_params['conv1_kernel_size'], 
                                                        stride=enc_arch_params['conv1_stride'], 
                                                        padding=enc_arch_params['conv1_padding'])
    
    pool1_output_size = calc_conv_or_pool_output_size(input_size=conv1_output_size, 
                                                        kernel_size=enc_arch_params['pool1_kernel_size'], 
                                                        stride=enc_arch_params['pool1_stride'], 
                                                        padding=enc_arch_params['pool1_padding'])
    
    if enc_arch_params['enc_multi_block'] == False:
        flat_size = pool1_output_size * enc_arch_params['conv1_ch_out']
        return int(flat_size)
    else:
        conv2_output_size = calc_conv_or_pool_output_size(input_size=pool1_output_size, 
                                                            kernel_size=enc_arch_params['conv2_kernel_size'], 
                                                            stride=enc_arch_params['conv2_stride'], 
                                                            padding=enc_arch_params['conv2_padding'])
        
        pool2_output_size = calc_conv_or_pool_output_size(input_size=conv2_output_size, 
                                                            kernel_size=enc_arch_params['pool2_kernel_size'], 
                                                            stride=enc_arch_params['pool2_stride'], 
                                                            padding=enc_arch_params['pool2_padding'])
            
        flat_size = pool2_output_size * enc_arch_params['conv2_ch_out']
        return int(flat_size)
    
def calc_convtranspose1d_output_size(input_size: int, kernel_size: int, stride: int, padding: int, output_padding: int=0, dilation: int=1):
    output_size = (input_size-1)*stride-2*padding+dilation*(kernel_size-1)+output_padding+1
    return output_size

def calculate_convtranspose1d_block_output_size(input_size: int, dec_arch_params: dict):
    
    convtranspose1_output_size = calc_convtranspose1d_output_size(input_size=input_size, 
                                                        kernel_size=dec_arch_params['convtr1_kernel_size'], 
                                                        stride=dec_arch_params['convtr1_stride'], 
                                                        padding=dec_arch_params['convtr1_padding'],
                                                        output_padding=dec_arch_params['convtr1_output_padding'],
                                                        dilation=dec_arch_params['convtr1_dilation'])
    
    if dec_arch_params['dec_multi_block'] == False:
        flat_size = convtranspose1_output_size * dec_arch_params['convtr1_ch_out']
        return int(flat_size)
    else:
        conv2_output_size = calc_convtranspose1d_output_size(input_size=convtranspose1_output_size, 
                                                            kernel_size=dec_arch_params['convtr2_kernel_size'], 
                                                            stride=dec_arch_params['convtr2_stride'], 
                                                            padding=dec_arch_params['convtr2_padding'],
                                                            output_padding=dec_arch_params['convtr2_output_padding'],
                                                            dilation=dec_arch_params['convtr2_dilation'])
            
        flat_size = conv2_output_size * dec_arch_params['convtr2_ch_out']
        return int(flat_size)


def calc_dilated_casual_conv1d_output_size(input_size: int, kernel_size: int, n_layer: int):
    max_dilation = 2**n_layer
    output_size = input_size-(kernel_size*max_dilation)+1
    return output_size

def calculate_tcn_block_output_size(input_size: int, enc_arch_params: dict):
    
    tcn_block1_output_size = calc_dilated_casual_conv1d_output_size(input_size=input_size, 
                                                                    kernel_size=enc_arch_params['tcn1_kernel_size'], 
                                                                    n_layer=enc_arch_params['tcn1_n_layer'])
    
    pool1_output_size = calc_conv_or_pool_output_size(input_size=tcn_block1_output_size, 
                                                        kernel_size=enc_arch_params['pool1_kernel_size'], 
                                                        stride=enc_arch_params['pool1_stride'], 
                                                        padding=enc_arch_params['pool1_padding'])
    
    flat_output_size = pool1_output_size * enc_arch_params['tcn1_ch_out']
    return int(flat_output_size)