# import os
# os.chdir(r'C:\Users\bonnyaigergo\Documents\GitHub\Deep-Temporal-Clustering')

from math import floor
from ae_models import EncoderConv1D, DecoderConv1D, EncoderTCN1D, EncoderRNN, DecoderRNN
from ae_models import Autoencoder, VariationalAutoEncoder, Discriminator, Generator
import optuna

def get_model_params(trial: optuna.Trial, model_type: str):
    
    encoder_conv1d_arch_params = {
       'conv1_ch_in': 1, 
       'conv1_ch_out': 4, # trial.suggest_int(name="conv1_ch_out", low=3, high=6, step=1),
       'conv1_kernel_size': 5,
       'conv1_stride': 1,
       'conv1_padding': 0,
       'pool1_kernel_size': 2,
       'pool1_stride': 2, # default value is kernel_size
       'pool1_padding': 0,
       'enc_multi_block': True, # trial.suggest_categorical("enc_multi_block", [True, False]),
       'conv2_ch_out': 8,
       'conv2_kernel_size': 5,
       'conv2_stride': 1,
       'conv2_padding': 0,
       'pool2_kernel_size': 2,
       'pool2_stride': 2, # default value is kernel_size
       'pool2_padding': 0
       }
         
    decoder_convtr1d_arch_params = {
        'convtr1_ch_in': 1,
        'convtr1_ch_out': 8, # trial.suggest_int(name="conv1_ch_out", low=3, high=8, step=1),
        'convtr1_kernel_size': 5,
        'convtr1_stride': 1,
        'convtr1_padding': 0,
        'convtr1_output_padding': 0,
        'convtr1_dilation': 1,
        'dec_multi_block': True, # trial.suggest_categorical("dec_multi_block", [True, False]),
        'convtr2_ch_out': 4,
        'convtr2_kernel_size': 10,
        'convtr2_stride': 1,
        'convtr2_padding': 0,
        'convtr2_output_padding': 0,
        'convtr2_dilation': 1
        }
    
    encoder_tcn_arch_params = {
        'enc_maxpool': True,
        'tcn1_ch_out': 5,
        'tcn1_n_layer': 4,
        'tcn1_kernel_size': 2,
        'tcn1_dropout_rate': 0.2,
        'pool1_kernel_size': 2,
        'pool1_stride': 2, # default value is kernel_size
        'pool1_padding': 0,
        'tcn2_ch_out': 5,
        'tcn2_n_layer': 4,
        'tcn2_kernel_size': 2,
        'tcn2_dropout_rate': 0.2,
        'pool2_kernel_size': 2,
        'pool2_stride': 2, # default value is kernel_size
        'pool2_padding': 0
        }
    
    encoder_rnn_arch_params = {
        'cell_type': 'GRU', # trial.suggest_categorical("enc_multi_block", ['GRU', 'LSTM']),
        'enc_multi_block': False,
        'hidden_size': 20,
        'layer1_n_layers': 5,
        'layer1_bidirectional': False,
        'layer1_dropout_rate': 0.2,
        'layer2_n_layers': 2,
        'layer2_bidirectional': False,
        'layer2_dropout_rate': 0.2,
        'before_fc_dropout': 0.4
        }
    
    decoder_rnn_arch_params = {
        'cell_type': 'GRU',  # trial.suggest_categorical("enc_multi_block", ['GRU', 'LSTM']),
        'dec_multi_block': True,
        'layer1_output_size': 15,
        'layer1_n_layers': 2,
        'layer1_bidirectional': False,
        'layer1_dropout_rate': 0.2,
        'layer2_output_size': 30,
        'layer2_n_layers': 2,
        'layer2_bidirectional': False,
        'layer2_dropout_rate': 0.2,
        'before_fc_dropout': 0.4
        }
    
    discriminator_arch_params = {
        'layer1_out': 128,
        'layer2_out': 64,
        'layer3_out': 32,
        'dropout': 0.3
        }
    
    gen_convtr1d_arch_params = {
        'random_input_size': 24,
        'convtr1_ch_in': 1,
        'convtr1_ch_out': 8, # trial.suggest_int(name="conv1_ch_out", low=3, high=8, step=1),
        'convtr1_kernel_size': 5,
        'convtr1_stride': 1,
        'convtr1_padding': 0,
        'convtr1_output_padding': 0,
        'convtr1_dilation': 1,
        'dec_multi_block': True, # trial.suggest_categorical("dec_multi_block", [True, False]),
        'convtr2_ch_out': 4,
        'convtr2_kernel_size': 10,
        'convtr2_stride': 1,
        'convtr2_padding': 0,
        'convtr2_output_padding': 0,
        'convtr2_dilation': 1
        }
    
    optim_params = {
        'learning_rate': trial.suggest_loguniform(name='learning_rate', low=1e-5, high=1e-1),
        'optimizer': trial.suggest_categorical(name="optimizer", choices=["Adam", "RMSprop", "SGD"]),
        'scheduler_step_size': trial.suggest_int(name="scheduler_step_size", low=15, high=75, step=15)
        }
    
         
    model_params_dict = {
        'Conv1D': {'encoder': EncoderConv1D,
                   'decoder': DecoderConv1D,
                   'autoencoder': Autoencoder,
                   'enc_arch_params': encoder_conv1d_arch_params,
                   'dec_arch_params': decoder_convtr1d_arch_params,
                   'optimization': optim_params
                   },
        'TCN':    {'encoder': EncoderTCN1D,
                   'decoder': DecoderConv1D,
                   'autoencoder': Autoencoder,
                   'enc_arch_params': encoder_tcn_arch_params,
                   'dec_arch_params': decoder_convtr1d_arch_params,
                   'optimization': optim_params
                   },
        'RNN':    {'encoder': EncoderRNN,
                   'decoder': DecoderRNN,
                   'autoencoder': Autoencoder,
                   'enc_arch_params': encoder_rnn_arch_params,
                   'dec_arch_params': decoder_rnn_arch_params,
                   'optimization': optim_params
                   },
        'VAE':    {'encoder': EncoderConv1D,
                   'decoder': DecoderConv1D,
                   'autoencoder': VariationalAutoEncoder,
                   'enc_arch_params': encoder_conv1d_arch_params,
                   'dec_arch_params': decoder_convtr1d_arch_params,
                   'optimization': optim_params
                   },
        'AAE':    {'encoder': EncoderConv1D,
                   'decoder': DecoderConv1D,
                   'autoencoder': Autoencoder,
                   'enc_arch_params': encoder_conv1d_arch_params,
                   'dec_arch_params': decoder_convtr1d_arch_params,
                   'discriminator': Discriminator,
                   'discr_arch_params': discriminator_arch_params,
                   'optimization': optim_params
                   },
        'GAN':    {'encoder': EncoderConv1D,
                   'decoder': DecoderConv1D,
                   'autoencoder': Autoencoder,
                   'enc_arch_params': encoder_conv1d_arch_params,
                   'dec_arch_params': decoder_convtr1d_arch_params,
                   'discriminator': Discriminator,
                   'discr_arch_params': discriminator_arch_params,
                   'generator': Generator,
                   'gen_arch_params': gen_convtr1d_arch_params,
                   'optimization': optim_params
                   }
        }
    
    return model_params_dict[model_type]
