# -*- coding: utf-8 -*-
"""
Experiment with Optuna and TensorBoard

optuna alternative: microsoft's nni
https://github.com/microsoft/nni/
ray tune:
https://debuggercafe.com/hyperparameter-tuning-with-pytorch-and-ray-tune/

good article:
https://www.analyticsvidhya.com/blog/2021/09/optimize-your-optimizations-using-optuna/
https://www.youtube.com/watch?v=b57ZBo9woLk&t=6s&ab_channel=JupyterCon

TODO: objective unit test:
https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-test-my-objective-functions

TODO: set GPU
https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-use-two-gpus-for-evaluating-two-trials-simultaneously
"""

# import os
# os.chdir(r'C:\Users\bonnyaigergo\Documents\GitHub\Deep-Temporal-Clustering')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import optuna

import itertools

import datasets
from utils import plot_loss, embedding_histogram, EarlyStopping
from datasets import CustomDataset
import model_params as mpar
from model_builder import build_ae_model, build_aae_model, build_gan_model
from trainer_tensorboard import train_ae_model, train_vae_model, train_aae_model, train_gan_model

# in case of VAE we have a bug and a TracerWarning, which is turned into silence
import warnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning) 

model_list = ['Conv1D', 'TCN', 'RNN', 'VAE', 'AAE', 'GAN']
DATASET = "Crop"
MODELTYPE = 'VAE'


# torchinfo.summary(model=model, input_size=(n_seq, n_timesteps, n_features))

# mlflow.pytorch.autolog()

def objective_ae_vae(trial: optuna.Trial, dataset: str=DATASET, model_type: str=MODELTYPE):
          
     model_params = mpar.get_model_params(trial, model_type)
     
     BATCHSIZE = 64 #trial.suggest_categorical("batch_size", [16, 32, 64])
     EPOCHS = 3
     EMBEDDING_SIZE = 16 #trial.suggest_int(name="batch_size", low=8, high=18, step=2)
     VAE_BETA = 0.8
     
     X, y, X_min, X_max = datasets.prepare_inputdata(dataset_name=dataset)

     # Train and validation sample split
     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=123)

     # Set parameters
     n_timesteps = X.shape[1]
     n_features = 1
     
     train_set = CustomDataset(X_train)
     val_set = CustomDataset(X_val)

     # Create data loader for pytorch
     train_dataloader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True)
     val_dataloader = DataLoader(val_set, batch_size=BATCHSIZE, shuffle=True)
     
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
     model = build_ae_model(n_timesteps, 
                             n_features, 
                             enc_arch_params=model_params['enc_arch_params'],
                             dec_arch_params=model_params['dec_arch_params'],
                             embedding_size=EMBEDDING_SIZE, 
                             batch_size=BATCHSIZE,
                             encoder_model=model_params['encoder'],
                             decoder_model=model_params['decoder'],
                             ae_model=model_params['autoencoder'])
                         
     model = model.to(device)
     
     # optimizer = getattr(optim, kwargs['optimizer'])(model.parameters(), lr=kwargs['learning_rate'])
     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
     scheduler = StepLR(optimizer, 
                        step_size=15, #kwargs['scheduler_step_size']
                        gamma=0.1)
     reconstruction_loss_fn = nn.MSELoss(reduction='mean').to(device)
     
     def vae_loss_fn(x_pred, x, mu, logvar, criterion, beta):    
         # reconstruction loss (pushing the points apart)
         loss_reconstruction = criterion(x_pred, x)
         # KL divergence loss (the relative entropy between two distributions a multivariate gaussian 
         # and a normal)
         # (enforce a radius of 1 in each direction + pushing the means towards zero)
         loss_KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
         loss = loss_reconstruction + beta*loss_KLD
         return loss, loss_reconstruction, loss_KLD
     
     if MODELTYPE=='VAE':
        target_metric = train_vae_model(model, 
                                        train_dataloader, 
                                        val_dataloader, 
                                        optimizer, 
                                        scheduler,
                                        reconstruction_loss_fn, 
                                        vae_loss_fn, 
                                        early_stopping=EarlyStopping(patience=5, delta=0, verbose=True),
                                        n_epochs=EPOCHS,
                                        checkpoint_path='runs/'+model_type+'_model'+str(trial.number)+'.pt',
                                        beta=VAE_BETA)            
     else:    
         target_metric = train_ae_model(model, 
                                        train_dataloader, 
                                        val_dataloader, 
                                        optimizer, 
                                        scheduler,
                                        reconstruction_loss_fn,
                                        early_stopping=EarlyStopping(patience=5, delta=0, verbose=True),
                                        n_epochs=EPOCHS,
                                        checkpoint_path='runs/'+model_type+'model'+str(trial.number)+'.pt')

     return target_metric

def objective_aae(trial: optuna.Trial, dataset: str=DATASET, model_type: str=MODELTYPE):
     
     model_params = mpar.get_model_params(trial, model_type='AAE')
     
     BATCHSIZE = 64 #trial.suggest_categorical("batch_size", [16, 32, 64])
     EPOCHS = 8
     EMBEDDING_SIZE = 16 #trial.suggest_int(name="batch_size", low=8, high=18, step=2)
     
     X, y, X_min, X_max = datasets.prepare_inputdata(dataset_name=dataset)

     # Train and validation sample split
     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=123)

     # Set parameters
     n_timesteps = X.shape[1]
     n_features = 1
     
     train_set = CustomDataset(X_train, train=True)
     val_set = CustomDataset(X_val, train=False)

     # Create data loader for pytorch
     train_dataloader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True)
     val_dataloader = DataLoader(val_set, batch_size=BATCHSIZE, shuffle=True)
     
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
     model, discriminator = build_aae_model(n_timesteps, 
                                             n_features, 
                                             enc_arch_params=model_params['enc_arch_params'],
                                             dec_arch_params=model_params['dec_arch_params'],
                                             discr_arch_params=model_params['discr_arch_params'],
                                             embedding_size=EMBEDDING_SIZE, 
                                             batch_size=BATCHSIZE,
                                             encoder_model=model_params['encoder'],
                                             decoder_model=model_params['decoder'],
                                             ae_model=model_params['autoencoder'],
                                             discriminator_model=model_params['discriminator']
                                             )
                                         
     model = model.to(device)
     discriminator = discriminator.to(device)
     
     optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
     scheduler_discriminator = StepLR(optimizer_discriminator, step_size=8, gamma=0.1)
     adversarial_loss_fn = nn.BCELoss().to(device)

     optimizer_generator = torch.optim.Adam(itertools.chain(model.encoder.parameters(), 
                                                            model.decoder.parameters(),
                                                            discriminator.parameters()), lr=1e-4)
     scheduler_generator = StepLR(optimizer_generator, step_size=8, gamma=0.1)
     reconstruction_loss_fn = nn.MSELoss(reduction='mean').to(device)
     
     target_metric = train_aae_model(model,
                                     discriminator,
                                     train_dataloader,
                                     val_dataloader,
                                     reconstruction_loss_fn,
                                     adversarial_loss_fn,
                                     optimizer_generator, 
                                     optimizer_discriminator,
                                     scheduler_generator,
                                     scheduler_discriminator,
                                     early_stopping=EarlyStopping(patience=5, delta=0, verbose=True),
                                     checkpoint_path='runs/'+model_type+'_model'+str(trial.number)+'.pt',
                                     n_epochs=EPOCHS)
     
     return target_metric

def objective_gan(trial: optuna.Trial, dataset: str=DATASET, model_type: str=MODELTYPE):
     
     model_params = mpar.get_model_params(trial, model_type='GAN')
     
     BATCHSIZE = 64 #trial.suggest_categorical("batch_size", [16, 32, 64])
     EPOCHS = 3
     EMBEDDING_SIZE = 16 #trial.suggest_int(name="batch_size", low=8, high=18, step=2)
     
     X, y, X_min, X_max = datasets.prepare_inputdata(dataset_name=dataset)

     # Train and validation sample split
     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=123)

     # Set parameters
     n_timesteps = X.shape[1]
     n_features = 1
     
     train_set = CustomDataset(X_train, train=True)
     val_set = CustomDataset(X_val, train=False)

     # Create data loader for pytorch
     train_dataloader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True)
     val_dataloader = DataLoader(val_set, batch_size=BATCHSIZE, shuffle=True)
     
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
     model, generator, discriminator = build_gan_model(n_timesteps, 
                                             n_features, 
                                             embedding_size=EMBEDDING_SIZE, 
                                             batch_size=BATCHSIZE,
                                             enc_arch_params=model_params['enc_arch_params'],
                                             dec_arch_params=model_params['dec_arch_params'],
                                             encoder_model=model_params['encoder'],
                                             decoder_model=model_params['decoder'],
                                             ae_model=model_params['autoencoder'],
                                             discriminator_model=model_params['discriminator'],
                                             discr_arch_params=model_params['discr_arch_params'],
                                             gen_arch_params=model_params['gen_arch_params'],
                                             generator_model=model_params['generator']
                                             )
                                         
     model = model.to(device)
     discriminator = discriminator.to(device)
     generator = generator.to(device)
     
     optimizer_ae = torch.optim.Adam(model.parameters(), lr=1e-4)
     scheduler_ae = StepLR(optimizer_ae, step_size=8, gamma=0.1)
     
     optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
     scheduler_discriminator = StepLR(optimizer_discriminator, step_size=8, gamma=0.1)
     adversarial_loss_fn = nn.BCELoss().to(device)

     optimizer_generator = torch.optim.Adam(itertools.chain(model.encoder.parameters(), model.decoder.parameters()), lr=1e-4)
     scheduler_generator = StepLR(optimizer_generator, step_size=8, gamma=0.1)
     reconstruction_loss_fn = nn.MSELoss(reduction='mean').to(device)
     
     target_metric = train_gan_model(model,
                                     generator,
                                     discriminator,
                                     train_dataloader,
                                     val_dataloader,
                                     reconstruction_loss_fn,
                                     adversarial_loss_fn,
                                     optimizer_ae,
                                     scheduler_ae,
                                     optimizer_discriminator,
                                     scheduler_discriminator,
                                     optimizer_generator,
                                     scheduler_generator,
                                     early_stopping=EarlyStopping(patience=5, delta=0, verbose=True),
                                     checkpoint_path='runs/'+model_type+'_model'+str(trial.number)+'.pt',
                                     n_epochs=EPOCHS)
     
     return target_metric
 

if __name__ == "__main__":

    # mlflow.set_experiment(experiment_name="TCN_experiment")
    # mlflow.end_run()
    # with mlflow.start_run(run_name="TCN_AE"):
        
    TRIALS = 1
        
    study = optuna.create_study(direction="minimize", 
                                sampler=optuna.samplers.TPESampler(), 
                                pruner=optuna.pruners.HyperbandPruner())
    if MODELTYPE == "AAE":
        study.optimize(objective_aae, n_trials=TRIALS)
    if MODELTYPE == "GAN":
        study.optimize(objective_gan, n_trials=TRIALS)
    else:
        study.optimize(objective_ae_vae, n_trials=TRIALS)

    
    # import ae_models
    # Load model parameters
    # trial_number = 0
    # checkpoint = torch.load('runs/VAE_model'+str(trial_number)+'.pt')
    # enc_arch_params = checkpoint['enc_arch_params']
    # dec_arch_params = checkpoint['dec_arch_params']
    # embedding_size = checkpoint['embedding_size']
    # batch_size = checkpoint['batch_size']
    # initialize VAE model
    # model = ae_models.AutoencoderVAE(ae_models.EncoderConv1D(n_timesteps, 
    #                                                         n_features=1, 
    #                                                         embedding_size=embedding_size, 
    #                                                         batch_size=batch_size,
    #                                                         enc_arch_params=enc_arch_params
    #                                                         ), 
    #                                  ae_models.DecoderConv1D(n_timesteps, 
    #                                                         n_features=1, 
    #                                                         embedding_size=embedding_size, 
    #                                                         batch_size=batch_size,
    #                                                         dec_arch_params=dec_arch_params
    #                                                         )
    #                                  )
    
    # optimizer = getattr(optim, 'Adam')(model.parameters(), lr=1e-3)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()
    # model.encoding(next(iter(train_dataloader)))
    

    
    # df = study.trials_dataframe().drop(['state','datetime_start','datetime_complete','duration','number'], axis=1)
    # df.tail(5)
    # df.sort_values(by=['value'])
    
    # best_trial = study.best_trial
    
    # for key, value in best_trial.params.items():
    #     print("{}: {}".format(key, value))  
    
    # import plotly
    # # Visualization
    # optuna.visualization.plot_intermediate_values(study)
    # optuna.visualization.plot_optimization_history(study)   
    # optuna.visualization.plot_parallel_coordinate(study)
    # optuna.visualization.plot_param_importances(study)
    # optuna.visualization.matplotlib.plot_param_importances(study)
    # optuna.visualization.plot_contour(study, params=['batch_size', 'lr'])
    
    # Fit
    # model, history, target_metric = train_model(
    #   model, 
    #   train_dataloader, 
    #   val_dataloader, 
    #   optimizer, 
    #   scheduler,
    #   loss_fn, 
    #   early_stopping=EarlyStopping(patience=5, delta=0, verbose=True),
    #   n_epochs=40,
    #   checkpoint_path=None
    # )
        
        # mlflow.log_param("batch_size", batch_size)
        # mlflow.log_param("dropout", dropout)
        # mlflow.log_metric("train_loss", train_loss)
        # mlflow.log_metric('val_loss', val_loss)
    # mlflow.end_run()
    
    # plot_loss(data=history, train='train', val="val", log_yscale=True)
    # plot_loss(data=history, train='loss_diff', val=None, log_yscale=False)
    
    # Anaconda terminal
    # mlflow ui
    # http://localhost:5000/
    
    # Tensorboard 
    # cmd code:
    # cd C:\Users\bonnyaigergo\Documents\GitHub\Deep-Temporal-Clustering
    # tensorboard --version
    # by running the following row the localhost is given, call it from the browser
    # tensorboard --logdir=runs
    # ls .\runs\
    # clear runs by deleting from runs folder

        
