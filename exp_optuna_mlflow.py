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

https://medium.com/@jerryhe.trader/using-mlflow-with-optuna-to-log-data-science-explorations-a-french-motor-claims-case-study-1ad3aa917d82
https://mlops.community/when-pytorch-meets-mlflow/
https://docs.databricks.com/_static/notebooks/mlflow/mlflow-pytorch-training.html
https://medium.com/swlh/pytorch-mlflow-optuna-experiment-tracking-and-hyperparameter-optimization-132778d6defc
https://madewithml.com/courses/mlops/experiment-tracking/
https://medium.com/analytics-vidhya/packaging-your-pytorch-model-using-mlflow-894d62dd8d3

"""

import os
os.chdir(r'C:\Users\bonnyaigergo\Documents\GitHub\Deep-Temporal-Clustering')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import itertools
import optuna
# MLFlow
import mlflow
import mlflow.tensorflow
import mlflow.pytorch
from mlflow import pytorch

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
MODEL_TYPE = 'Conv1D'

TRACKING_URI = 'http://127.0.0.1:5000'
# TRACKING_URI = './mlruns'
# mlflow.list_experiments()
# mlflow.delete_experiment('5')
# mlflow.tracking.restore_experiment('1')
# mlflow gc experiments --backend-store-uri ‘http://127.0.0.1:5000’ --experiments -ids 1,2,3,4

# torchinfo.summary(model=model, input_size=(n_seq, n_timesteps, n_features))

def objective_ae(trial: optuna.Trial, dataset: str=DATASET, model_type: str=MODEL_TYPE):
    
     # mlflow.set_tracking_uri(TRACKING_URI)
     mlflow.set_experiment(experiment_name=dataset+"_"+model_type+"_exp")
     
     with mlflow.start_run(run_name=DATASET+"_"+MODEL_TYPE+str(trial.number)):
          
         model_params = mpar.get_model_params(trial, model_type)
         
         BATCH_SIZE = 64 #trial.suggest_categorical("batch_size", [16, 32, 64])
         N_EPOCH = 3
         EMBEDDING_SIZE = 16 #trial.suggest_int(name="embedding_size", low=8, high=18, step=2)
         
         X, y, X_min, X_max = datasets.prepare_inputdata(dataset_name=dataset)
    
         # Train and validation sample split
         X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=123)
    
         # Set parameters
         n_timesteps = X.shape[1]
         n_features = 1
         
         train_set = CustomDataset(X_train)
         val_set = CustomDataset(X_val)
    
         # Create data loader for pytorch
         train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
         val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
         
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         
         model = build_ae_model(n_timesteps, 
                                 n_features, 
                                 enc_arch_params=model_params['enc_arch_params'],
                                 dec_arch_params=model_params['dec_arch_params'],
                                 embedding_size=EMBEDDING_SIZE, 
                                 batch_size=BATCH_SIZE,
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
         
         checkpoint_path='runs/'+model_type+'model'+str(trial.number)+'.pt'
         
         history = dict(train=[], val=[], loss_diff=[])
         
         early_stopping=EarlyStopping(patience=5, delta=0, verbose=True)
         
         for epoch in tqdm(range(1, N_EPOCH + 1)):
           
                train_loss_batches = []
                val_loss_batches = []
                
                # Training
                model = model.train()
                for i, seq_true in enumerate(train_dataloader):
                    optimizer.zero_grad()
                    seq_true = seq_true.to(device)
                    seq_pred = model(seq_true)
                    loss = reconstruction_loss_fn(seq_pred, seq_true)
                    loss.backward()
                    optimizer.step()
                    train_loss_batches.append(loss.item())
                                
               # Validation
                model = model.eval()
                with torch.no_grad():
                    for i, seq_true in enumerate(val_dataloader):
                        seq_true = seq_true.to(device)
                        seq_pred = model(seq_true)
                        loss = reconstruction_loss_fn(seq_pred, seq_true)
                        val_loss_batches.append(loss.item())
                
                train_loss_epoch = np.mean(train_loss_batches)
                val_loss_epoch = np.mean(val_loss_batches)
                loss_diff_epoch = np.abs(train_loss_epoch - val_loss_epoch)
                
                mlflow.log_params(trial.params)
                mlflow.log_metric("train_loss", train_loss_epoch)
                mlflow.log_metric('val_loss', val_loss_epoch)
                
                history['train'].append(train_loss_epoch)
                history['val'].append(val_loss_epoch)
                history['loss_diff'].append(loss_diff_epoch)
            
                print(f'Epoch {epoch}: train loss {train_loss_epoch} val loss {val_loss_epoch}')
                
                early_stopping(val_loss_epoch, model)
                if early_stopping.meet_criterion:
                    break
                
                # Learning rate decay
                scheduler.step()
                
         torch.save({'model_state_dict': early_stopping.best_model_weights,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'enc_arch_params': model.encoder.enc_arch_params,
                    'dec_arch_params': model.decoder.dec_arch_params,               
                    'batch_size': model.encoder.batch_size,
                    'embedding_size': model.decoder.embedding_size,
                    'n_timesteps': model.encoder.n_timesteps
                    },
                   checkpoint_path)
        
         target_metric = np.min(history['train'] + history['loss_diff'])

     return target_metric
 

def objective_vae(trial: optuna.Trial, dataset: str=DATASET, model_type: str=MODEL_TYPE):
    
     mlflow.set_tracking_uri('http://127.0.0.1:5000')
     mlflow.set_experiment(experiment_name=dataset+"_"+model_type+"_experiment")
     
     with mlflow.start_run(run_name=DATASET+MODEL_TYPE):
          
         model_params = mpar.get_model_params(trial, model_type)
         
         BATCH_SIZE = 64 #trial.suggest_categorical("batch_size", [16, 32, 64])
         N_EPOCH = 3
         EMBEDDING_SIZE = 16 #trial.suggest_int(name="embedding_size", low=8, high=18, step=2)
         VAE_BETA = 0.8
         
         X, y, X_min, X_max = datasets.prepare_inputdata(dataset_name=dataset)
    
         # Train and validation sample split
         X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=123)
    
         # Set parameters
         n_timesteps = X.shape[1]
         n_features = 1
         
         train_set = CustomDataset(X_train, train=True)
         val_set = CustomDataset(X_val, train=False)
    
         # Create data loader for pytorch
         train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
         val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
         
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         
         model = build_ae_model(n_timesteps, 
                                 n_features, 
                                 enc_arch_params=model_params['enc_arch_params'],
                                 dec_arch_params=model_params['dec_arch_params'],
                                 embedding_size=EMBEDDING_SIZE, 
                                 batch_size=BATCH_SIZE,
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
         
         checkpoint_path='runs/'+model_type+'model'+str(trial.number)+'.pt'
         
         history = dict(train=[], val=[], loss_diff=[])
         
         early_stopping=EarlyStopping(patience=5, delta=0, verbose=True)
         
         for epoch in tqdm(range(1, N_EPOCH + 1)):

              train_loss_batches = []
              val_loss_batches = []
              
              train_KLD_batches = []
              val_KLD_batches = []
              
              train_rec_loss_batches =  []
              val_rec_loss_batches = []
            
              # Training step
              model = model.train()
              for i, seq_true in enumerate(train_dataloader):
                  optimizer.zero_grad()
                  seq_true = seq_true.to(device)
                  mu, logvar, _, seq_pred = model(seq_true)
                  loss, loss_reconstruction, loss_KLD = vae_loss_fn(seq_pred, 
                                                                    seq_true, 
                                                                    mu, 
                                                                    logvar, 
                                                                    reconstruction_loss_fn,
                                                                    beta=VAE_BETA)
                  loss.backward()
                  optimizer.step()
                  train_loss_batches.append(loss.item())
                  train_KLD_batches.append(loss_KLD.item())
                  train_rec_loss_batches.append(loss_reconstruction.item())
          
              model = model.eval()
              
              # Validation step
              with torch.no_grad():
                  for i, seq_true in enumerate(val_dataloader):
                      seq_true = seq_true.to(device)
                      mu, logvar, _, seq_pred = model(seq_true)
                      loss, loss_reconstruction, loss_KLD = vae_loss_fn(seq_pred, 
                                                                        seq_true, 
                                                                        mu, 
                                                                        logvar, 
                                                                        reconstruction_loss_fn,
                                                                        beta=VAE_BETA)
                      val_loss_batches.append(loss.item())
                      val_KLD_batches.append(loss_KLD.item())
                      val_rec_loss_batches.append(loss_reconstruction.item())
          
              train_loss_epoch = np.mean(train_loss_batches)
              val_loss_epoch = np.mean(val_loss_batches)
              loss_diff_epoch = np.abs(train_loss_epoch - val_loss_epoch)
              
              train_KLD_loss_epoch = np.mean(train_KLD_batches)
              val_KLD_loss_epoch = np.mean(val_KLD_batches)
              
              train_rec_loss_epoch = np.mean(train_rec_loss_batches)
              val_rec_loss_epoch = np.mean(val_rec_loss_batches)
              
              history['train'].append(train_loss_epoch)
              history['val'].append(val_loss_epoch)
              history['loss_diff'].append(loss_diff_epoch)
              
              mlflow.log_params(trial.params)
              mlflow.log_metric("train_loss", train_loss_epoch)
              mlflow.log_metric('val_loss', val_loss_epoch)
              mlflow.log_metric('train_KLD_loss', train_KLD_loss_epoch)
              mlflow.log_metric('val_KLD_loss', val_KLD_loss_epoch)
              mlflow.log_metric('train_rec_loss', train_rec_loss_epoch)
              mlflow.log_metric('val_rec_loss', val_rec_loss_epoch)
             
              print(f'Epoch {epoch}: \n\
                      train loss {train_loss_epoch} \n\
                      val loss {val_loss_epoch} \n\
                      train KLD loss {train_KLD_loss_epoch} \n\
                      val KLD loss {val_KLD_loss_epoch} \n\
                      train rec loss {train_rec_loss_epoch} \n\
                      val rec loss {val_rec_loss_epoch}')
                    
              early_stopping(val_loss_epoch, model)
              if early_stopping.meet_criterion:
                  break
              
              # Learning rate decay
              scheduler.step()
        
        # TODO: set multiple metric
         torch.save({'model_state_dict': early_stopping.best_model_weights,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'enc_arch_params': model.encoder.enc_arch_params,
                    'dec_arch_params': model.decoder.dec_arch_params,               
                    'batch_size': model.encoder.batch_size,
                    'embedding_size': model.decoder.embedding_size,
                    'n_timesteps': model.encoder.n_timesteps
                    },
                   checkpoint_path)
        
         target_metric = np.min(history['train'] + history['loss_diff'])
                
     return target_metric
 


def objective_aae(trial: optuna.Trial, dataset: str = DATASET, model_type: str = MODEL_TYPE):
    
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment(experiment_name=dataset + "_" + model_type + "_experiment")

    with mlflow.start_run(run_name=DATASET + MODEL_TYPE):

        model_params = mpar.get_model_params(trial, model_type='AAE')

        BATCH_SIZE = 64  # trial.suggest_categorical("batch_size", [16, 32, 64])
        N_EPOCH = 8
        EMBEDDING_SIZE = 16  # trial.suggest_int(name="batch_size", low=8, high=18, step=2)

        X, y, X_min, X_max = datasets.prepare_inputdata(dataset_name=dataset)

        # Train and validation sample split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=123)

        # Set parameters
        n_timesteps = X.shape[1]
        n_features = 1

        train_set = CustomDataset(X_train, train=True)
        val_set = CustomDataset(X_val, train=False)

        # Create data loader for pytorch
        train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model, discriminator = build_aae_model(n_timesteps,
                                               n_features,
                                               enc_arch_params=model_params['enc_arch_params'],
                                               dec_arch_params=model_params['dec_arch_params'],
                                               discr_arch_params=model_params['discr_arch_params'],
                                               embedding_size=EMBEDDING_SIZE,
                                               batch_size=BATCH_SIZE,
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

        checkpoint_path = 'runs/' + model_type + 'model' + str(trial.number) + '.pt'

        history = dict(train_rec=[], val_rec=[], train_discr=[], val_discr=[],
                       generator=[], rec_loss_diff=[], discr_loss_diff=[])

        early_stopping = EarlyStopping(patience=5, delta=0, verbose=True)

        for epoch in tqdm(range(1, N_EPOCH + 1)):
            
            generator_loss_batches = []
            train_discr_loss_batches = []
            val_discr_loss_batches = []
            train_rec_loss_batches = []
            val_rec_loss_batches = []

            # Training step
            for i, real_samples in enumerate(train_dataloader):
                # real_samples = next(iter(train_dataloader))
                batch_size = real_samples.shape[0]

                # Data for training the discriminator
                real = torch.ones((batch_size, 1))
                fake = torch.zeros((batch_size, 1))

                # 1) reconstruction + generator loss
                optimizer_generator.zero_grad()
                real_latent = model.encoder(real_samples)
                real_decoded = model.decoder(real_latent)
                validity_real_latent = discriminator(real_latent)

                rec_loss = reconstruction_loss_fn(real_decoded, real_samples)
                generator_loss = 0.001 * adversarial_loss_fn(validity_real_latent, real) + 0.999 * rec_loss
                generator_loss.backward()
                optimizer_generator.step()

                # 2) discriminator loss
                optimizer_discriminator.zero_grad()
                fake_latent = torch.randn((batch_size, model.encoder.embedding_size))
                fake_loss = adversarial_loss_fn(discriminator(fake_latent), fake)
                real_loss = adversarial_loss_fn(discriminator(real_latent.detach()), real)
                discriminator_loss = 0.5 * (real_loss + fake_loss)
                discriminator_loss.backward()
                optimizer_discriminator.step()

                generator_loss_batches.append(generator_loss.item())
                train_discr_loss_batches.append(discriminator_loss.item())
                train_rec_loss_batches.append(rec_loss.item())

            # Validation step
            with torch.no_grad():
                for i, real_sample in enumerate(val_dataloader):
                    # real_sample = next(iter(val_dataloader))
                    real_sample = real_sample.to(device)
                    real_latent = model.encoder(real_sample)
                    real_decoded = model.decoder(real_latent)
                    rec_loss = reconstruction_loss_fn(real_decoded, real_sample)

                    fake_latent = torch.randn((batch_size, model.encoder.embedding_size))
                    fake_loss = adversarial_loss_fn(discriminator(fake_latent), fake)
                    real_loss = adversarial_loss_fn(discriminator(real_latent), real)
                    discriminator_loss = 0.5 * (real_loss + fake_loss)
                    val_discr_loss_batches.append(discriminator_loss.item())
                    val_rec_loss_batches.append(rec_loss.item())

            train_discr_loss_epoch = np.mean(train_discr_loss_batches)
            val_discr_loss_epoch = np.mean(val_discr_loss_batches)
            generator_loss_epoch = np.mean(generator_loss_batches)
            train_rec_loss_epoch = np.mean(train_rec_loss_batches)
            val_rec_loss_epoch = np.mean(val_rec_loss_batches)
            rec_loss_diff_epoch = np.abs(train_rec_loss_epoch - val_rec_loss_epoch)
            discr_loss_diff_epoch = np.abs(train_discr_loss_epoch - val_discr_loss_epoch)
            history['train_rec'].append(train_rec_loss_epoch)
            history['val_rec'].append(val_rec_loss_epoch)
            history['train_discr'].append(train_discr_loss_epoch)
            history['val_discr'].append(val_discr_loss_epoch)
            history['generator'].append(generator_loss_epoch)
            history['rec_loss_diff'].append(rec_loss_diff_epoch)
            history['discr_loss_diff'].append(discr_loss_diff_epoch)
            
            mlflow.log_params(trial.params)
            mlflow.log_metric('train_rec_loss', train_rec_loss_epoch)
            mlflow.log_metric('val_rec_loss', val_rec_loss_epoch)
            mlflow.log_metric("train_discr_loss", train_discr_loss_epoch)
            mlflow.log_metric('val_discr_loss', val_discr_loss_epoch)
            mlflow.log_metric('generator_loss', generator_loss_epoch)

            # Learning rate decay
            scheduler_discriminator.step()
            scheduler_generator.step()

            print(f'Epoch {epoch}: \n\
                       Train rec loss {train_rec_loss_epoch} \n\
                       Val rec loss {val_rec_loss_epoch} \n\
                       Generator loss {generator_loss_epoch} \n\
                       Train discr loss {train_discr_loss_epoch} \n\
                       Val discr loss {val_discr_loss_epoch}')

            # TODO: refactor early stopping for tracking multiple loss
            early_stopping(val_rec_loss_epoch, model)
            if early_stopping.meet_criterion:
                break

        # TODO: set multiple metric
        torch.save({'model_state_dict': early_stopping.best_model_weights,
                    'optim_gen_state_dict': optimizer_generator.state_dict(),
                    'enc_arch_params': model.encoder.enc_arch_params,
                    'dec_arch_params': model.decoder.dec_arch_params,
                    'discr_state_dict': discriminator.state_dict(),
                    'discr_arch_params': discriminator.discr_arch_params,
                    'optim_discr_state_dict': optimizer_discriminator.state_dict(),
                    'batch_size': model.encoder.batch_size,
                    'embedding_size': model.decoder.embedding_size,
                    'n_timesteps': model.encoder.n_timesteps
                    },
                   checkpoint_path)

        target_metric = np.min(history['train_rec'] + history['rec_loss_diff'])

    return target_metric

def objective_gan(trial: optuna.Trial, dataset: str=DATASET, model_type: str=MODEL_TYPE):
     
     mlflow.set_tracking_uri('http://127.0.0.1:5000')
     mlflow.set_experiment(experiment_name=dataset + "_" + model_type + "_experiment")

     with mlflow.start_run(run_name=DATASET + MODEL_TYPE):
         model_params = mpar.get_model_params(trial, model_type='GAN')
         
         BATCH_SIZE = 64 #trial.suggest_categorical("batch_size", [16, 32, 64])
         N_EPOCH = 3
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
         train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
         val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
         
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         
         model, generator, discriminator = build_gan_model(n_timesteps, 
                                                 n_features, 
                                                 embedding_size=EMBEDDING_SIZE, 
                                                 batch_size=BATCH_SIZE,
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
         
         checkpoint_path = 'runs/' + model_type + 'model' + str(trial.number) + '.pt'

         history = dict(train_rec=[], val_rec=[], train_discr=[], val_discr=[], 
                       train_generator=[], val_generator=[], rec_loss_diff=[], discr_loss_diff=[])

         early_stopping = EarlyStopping(patience=5, delta=0, verbose=True)
         
         for epoch in tqdm(range(1, N_EPOCH + 1)):
             
             train_gen_loss_batches = []
             val_gen_loss_batches = [] 
             train_discr_loss_batches = []
             val_discr_loss_batches = []
             train_rec_loss_batches = []
             val_rec_loss_batches = []

             # Training step
             for i, real_samples in enumerate(train_dataloader):
                 # real_samples = next(iter(train_dataloader))
                 batch_size = real_samples.shape[0]
         
                 # Data for training the discriminator
                 real = torch.ones((batch_size, 1))
                 fake = torch.zeros((batch_size, 1))
         
                 # 1) reconstruction + generator loss
                 optimizer_generator.zero_grad()
                 real_latent = model.encoder(real_samples)
                 real_decoded = model.decoder(real_latent)
                 validity_real_latent = discriminator(real_latent)
                 
                 rec_loss = reconstruction_loss_fn(real_decoded, real_samples)
                 ae_loss = 0.001*adversarial_loss_fn(validity_real_latent, real) + 0.999*rec_loss
                 ae_loss.backward()
                 optimizer_ae.step()
                 
                 # 2) generator loss
                 fake_init = torch.randn((batch_size, generator.random_input_size))
                 fake_samples = generator(fake_init)
                 # There is no training on the encoder
                 with torch.no_grad():
                     fake_latent = model.encoder(fake_samples)
                 generator_loss = adversarial_loss_fn(discriminator(fake_latent), real)
                 generator_loss.backward()
                 optimizer_generator.step()
                 
                 # 3) discriminator loss
                 optimizer_discriminator.zero_grad()
                 fake_init = torch.randn((batch_size, generator.random_input_size))
                 fake_samples = generator(fake_init)
                 fake_latent = model.encoder(fake_samples)
                 fake_loss = adversarial_loss_fn(discriminator(fake_latent), fake)
                 real_loss = adversarial_loss_fn(discriminator(real_latent.detach()), real)
                 discriminator_loss = 0.5*(real_loss + fake_loss)
                 discriminator_loss.backward()
                 optimizer_discriminator.step()
         
                 train_gen_loss_batches.append(generator_loss.item())
                 train_discr_loss_batches.append(discriminator_loss.item())
                 train_rec_loss_batches.append(rec_loss.item())

             # Validation step
             with torch.no_grad():
                 for i, real_sample in enumerate(val_dataloader):
                     # real_sample = next(iter(val_dataloader))
                     real_sample = real_sample.to(device)
                     real_latent = model.encoder(real_sample)
                     real_decoded = model.decoder(real_latent)
                     rec_loss = reconstruction_loss_fn(real_decoded, real_sample)
                     
                     fake_init = torch.randn((batch_size, generator.random_input_size))
                     fake_samples = generator(fake_init)
                     fake_latent = model.encoder(fake_samples)
                     generator_loss = adversarial_loss_fn(discriminator(fake_latent), real)
                     
                     fake_loss = adversarial_loss_fn(discriminator(fake_latent), fake)
                     real_loss = adversarial_loss_fn(discriminator(real_latent), real)
                     discriminator_loss = 0.5 * (real_loss + fake_loss)

                     val_discr_loss_batches.append(discriminator_loss.item())       
                     val_rec_loss_batches.append(rec_loss.item())
                     val_gen_loss_batches.append(generator_loss.item())

             train_discr_loss_epoch = np.mean(train_discr_loss_batches)
             val_discr_loss_epoch = np.mean(val_discr_loss_batches)
             train_gen_loss_epoch = np.mean(train_gen_loss_batches)
             val_gen_loss_epoch = np.mean(val_gen_loss_batches)
             train_rec_loss_epoch = np.mean(train_rec_loss_batches)
             val_rec_loss_epoch = np.mean(val_rec_loss_batches)
             rec_loss_diff_epoch = np.abs(train_rec_loss_epoch - val_rec_loss_epoch)
             discr_loss_diff_epoch = np.abs(train_discr_loss_epoch - val_discr_loss_epoch)
             history['train_rec'].append(train_rec_loss_epoch)
             history['val_rec'].append(val_rec_loss_epoch)
             history['train_discr'].append(train_discr_loss_epoch)
             history['val_discr'].append(val_discr_loss_epoch)
             history['train_generator'].append(train_gen_loss_epoch)
             history['val_generator'].append(val_gen_loss_epoch)
             history['rec_loss_diff'].append(rec_loss_diff_epoch)
             history['discr_loss_diff'].append(discr_loss_diff_epoch)
             
             mlflow.log_params(trial.params)
             mlflow.log_metric('train_rec_loss', train_rec_loss_epoch)
             mlflow.log_metric('val_rec_loss', val_rec_loss_epoch)
             mlflow.log_metric("train_discr_loss", train_discr_loss_epoch)
             mlflow.log_metric('val_discr_loss', val_discr_loss_epoch)
             mlflow.log_metric('train_generator_loss', train_gen_loss_epoch)
             mlflow.log_metric('val_generator_loss', val_gen_loss_epoch)
         
             # Learning rate decay
             scheduler_discriminator.step()
             scheduler_generator.step()
             scheduler_ae.step()
             
             print(f'Epoch {epoch}: \n\
                     Train rec loss {train_rec_loss_epoch} \n\
                     Val rec loss {val_rec_loss_epoch} \n\
                     Train gen loss {train_gen_loss_epoch} \n\
                     Train discr loss {train_discr_loss_epoch} \n\
                     Val discr loss {val_discr_loss_epoch}')
                     
             # TODO: refactor early stopping for tracking multiple loss
             early_stopping(val_rec_loss_epoch, model)
             if early_stopping.meet_criterion:
                 break
                     
         # TODO: set multiple metric
         torch.save({'model_state_dict': early_stopping.best_model_weights,
                     'optim_gen_state_dict': optimizer_generator.state_dict(),
                     'enc_arch_params': model.encoder.enc_arch_params,
                     'dec_arch_params': model.decoder.dec_arch_params,
                     'discr_state_dict': discriminator.state_dict(),
                     'discr_arch_params': discriminator.discr_arch_params,
                     'optim_discr_state_dict': optimizer_discriminator.state_dict(),
                     'batch_size': model.encoder.batch_size,
                     'embedding_size': model.decoder.embedding_size,
                     'n_timesteps': model.encoder.n_timesteps
                     },
                    checkpoint_path)

         target_metric = np.min(history['train_rec'] + history['rec_loss_diff'])
          
     return target_metric
 

if __name__ == "__main__":

    # mlflow.set_experiment(experiment_name="TCN_experiment")
    # mlflow.end_run()
    # with mlflow.start_run(run_name="TCN_AE"):
        
    N_TRIALS = 1
        
    study = optuna.create_study(direction="minimize", 
                                sampler=optuna.samplers.TPESampler(), 
                                pruner=optuna.pruners.HyperbandPruner())
    if MODEL_TYPE == "AAE":
        study.optimize(objective_aae, n_trials=N_TRIALS)
    if MODEL_TYPE == "GAN":
        study.optimize(objective_gan, n_trials=N_TRIALS)
    if MODEL_TYPE == "VAE":
        study.optimize(objective_vae, n_trials=N_TRIALS)
    else:
        study.optimize(objective_ae, n_trials=N_TRIALS)

    mlflow.end_run()
    
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
    # delete experiment:
    # mlflow experiments delete --experiment-id Crop_TCN_experiment
    
    # Tensorboard 
    # cmd code:
    # cd C:\Users\bonnyaigergo\Documents\GitHub\Deep-Temporal-Clustering
    # tensorboard --version
    # by running the following row the localhost is given, call it from the browser
    # tensorboard --logdir=runs
    # ls .\runs\
    # clear runs by deleting from runs folder
        

