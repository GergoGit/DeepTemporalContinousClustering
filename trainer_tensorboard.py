# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 03:24:07 2022

@author: bonnyaigergo
"""

# import os
# os.chdir(r'C:\Users\bonnyaigergo\Documents\GitHub\Deep-Temporal-Clustering')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from tqdm import tqdm
from typing import Callable, Type
from utils import EarlyStopping


def train_ae_model(model: nn.Module, 
                   train_dataloader: DataLoader, 
                   val_dataloader: DataLoader, 
                   optimizer: optim.Optimizer, 
                   scheduler: optim.lr_scheduler,
                   reconstruction_loss_fn: Callable, 
                   early_stopping: Type[EarlyStopping],
                   n_epochs: int,
                   checkpoint_path: str):

    # history for plotting
    history = dict(train=[], val=[], loss_diff=[])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TensorBoard
    tb = SummaryWriter()
    
    # tqdm provide a good looking progress bar
    for epoch in tqdm(range(1, n_epochs + 1)):
          model = model.train()
      
          train_loss_batches = []
          val_loss_batches = []
      
          # Training step
          for i, seq_true in enumerate(train_dataloader):
              if i == 0:
                  tb.add_graph(model=model, input_to_model=seq_true)
              optimizer.zero_grad()
              seq_true = seq_true.to(device)
              seq_pred = model(seq_true)
              loss = reconstruction_loss_fn(seq_pred, seq_true)
              loss.backward()
              optimizer.step()
              train_loss_batches.append(loss.item())
      
          model = model.eval()
          
          # Validation step
          with torch.no_grad():
              for i, seq_true in enumerate(val_dataloader):
                  seq_true = seq_true.to(device)
                  seq_pred = model(seq_true)
                  loss = reconstruction_loss_fn(seq_pred, seq_true)
                  val_loss_batches.append(loss.item())
      
          train_loss_epoch = np.mean(train_loss_batches)
          val_loss_epoch = np.mean(val_loss_batches)
          loss_diff_epoch = np.abs(train_loss_epoch - val_loss_epoch)
          
          history['train'].append(train_loss_epoch)
          history['val'].append(val_loss_epoch)
          history['loss_diff'].append(loss_diff_epoch)
          
          tb.add_scalar(tag='train_loss', scalar_value=train_loss_epoch, global_step=epoch)
          tb.add_scalar(tag='val_loss', scalar_value=val_loss_epoch, global_step=epoch)
          tb.add_scalar(tag='loss_diff', scalar_value=loss_diff_epoch, global_step=epoch)
          
          print(f'Epoch {epoch}: train loss {train_loss_epoch} val loss {val_loss_epoch}')
          
          early_stopping(val_loss_epoch, model)
          if early_stopping.meet_criterion:
              break
          
          # Learning rate decay
          scheduler.step()
                       
    tb.add_hparams({'batch_size': model.encoder.batch_size},
                   {'train_loss': history['train'][-1],
                    'val_loss': history['val'][-1],
                    'loss_diff': history['loss_diff'][-1],
                    }
                   )

    tb.close()
    # TODO: set multiple metric
    torch.save({'model_state_dict': early_stopping.best_model_weights,
                'optimizer_state_dict': optimizer.state_dict(),
                'enc_arch_params': model.encoder.enc_arch_params,
                'dec_arch_params': model.decoder.dec_arch_params,               
                'batch_size': model.encoder.batch_size,
                'embedding_size': model.decoder.embedding_size
                },
               checkpoint_path)
    
    target_metric = np.min(history['train'] + history['loss_diff'])
    
    return target_metric

def train_vae_model(model: nn.Module, 
                    train_dataloader: DataLoader, 
                    val_dataloader: DataLoader, 
                    optimizer: optim.Optimizer, 
                    scheduler: optim.lr_scheduler,
                    reconstruction_loss_fn: Callable, 
                    vae_loss_fn: Callable,
                    early_stopping: Type[EarlyStopping],
                    n_epochs: int,
                    checkpoint_path: str,
                    beta: float):

    # history for plotting
    history = dict(train=[], val=[], loss_diff=[])
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TensorBoard
    tb = SummaryWriter()
  
    # tqdm provide a good looking progress bar
    for epoch in tqdm(range(1, n_epochs + 1)):
          model = model.train()
          
          train_loss_batches = []
          val_loss_batches = []
          
          train_KLD_batches = []
          val_KLD_batches = []
          
          train_rec_loss_batches =  []
          val_rec_loss_batches = []
        
          # Training step
          for i, seq_true in enumerate(train_dataloader):
              if i == 0:
                  tb.add_graph(model=model, input_to_model=seq_true)
              optimizer.zero_grad()
              seq_true = seq_true.to(device)
              mu, logvar, _, seq_pred = model(seq_true)
              loss, loss_reconstruction, loss_KLD = vae_loss_fn(seq_pred, 
                                                                seq_true, 
                                                                mu, 
                                                                logvar, 
                                                                reconstruction_loss_fn,
                                                                beta)
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
                                                                    beta)
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
          
          tb.add_scalar(tag='train_loss', scalar_value=train_loss_epoch)
          tb.add_scalar(tag='val_loss', scalar_value=val_loss_epoch)
          tb.add_scalar(tag='train_KLD_loss', scalar_value=train_KLD_loss_epoch)
          tb.add_scalar(tag='val_KLD_loss', scalar_value=val_KLD_loss_epoch)
          tb.add_scalar(tag='train_rec_loss', scalar_value=train_rec_loss_epoch)
          tb.add_scalar(tag='val_rec_loss', scalar_value=val_rec_loss_epoch)
          
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
                       
    tb.add_hparams({'batch_size': model.encoder.batch_size},
                   {'train_loss': history['train'][-1],
                    'val_loss': history['val'][-1],
                    'loss_diff': history['loss_diff'][-1],
                    }
                   )

    tb.close()
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



def train_aae_model(model: nn.Module,
                    discriminator: nn.Module,
                    train_dataloader: DataLoader,
                    val_dataloader: DataLoader,
                    reconstruction_loss_fn: Callable,
                    adversarial_loss_fn: Callable,
                    optimizer_generator: optim.Optimizer, 
                    optimizer_discriminator: optim.Optimizer,
                    scheduler_generator: optim.lr_scheduler,
                    scheduler_discriminator: optim.lr_scheduler,
                    early_stopping: Type[EarlyStopping],
                    checkpoint_path: str,
                    n_epochs: int):
    
    # TODO: check for checkpoint folder
    # history for plotting
    history = dict(train_rec=[], val_rec=[], train_discr=[], val_discr=[], 
                   generator=[], rec_loss_diff=[], discr_loss_diff=[])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TensorBoard
    tb = SummaryWriter()

    for epoch in tqdm(range(1, n_epochs + 1)):
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
            generator_loss = 0.001*adversarial_loss_fn(validity_real_latent, real) + 0.999*rec_loss
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
                
    tb.add_hparams({'batch_size': model.encoder.batch_size},
                   {'train_rec_loss': history['train_rec'][-1],
                    'val_rec_loss': history['val_rec'][-1],
                    'rec_loss_diff': history['rec_loss_diff'][-1],
                    }
                   )
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

def train_gan_model(model: nn.Module,
                    generator: nn.Module,
                    discriminator: nn.Module,
                    train_dataloader: DataLoader,
                    val_dataloader: DataLoader,
                    reconstruction_loss_fn: Callable,
                    adversarial_loss_fn: Callable,
                    optimizer_ae: optim.Optimizer,
                    scheduler_ae: optim.lr_scheduler,
                    optimizer_discriminator: optim.Optimizer,
                    scheduler_discriminator: optim.lr_scheduler,
                    optimizer_generator: optim.Optimizer,
                    scheduler_generator: optim.lr_scheduler,
                    early_stopping: Type[EarlyStopping],
                    checkpoint_path: str,
                    n_epochs: int):
    
    # TODO: check for checkpoint folder
    # history for plotting
    history = dict(train_rec=[], val_rec=[], train_discr=[], val_discr=[], 
                   train_generator=[], val_generator=[], rec_loss_diff=[], discr_loss_diff=[])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TensorBoard
    tb = SummaryWriter()

    for epoch in tqdm(range(1, n_epochs + 1)):
        
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
                
    tb.add_hparams({'batch_size': model.encoder.batch_size},
                   {'train_rec_loss': history['train_rec'][-1],
                    'val_rec_loss': history['val_rec'][-1],
                    'rec_loss_diff': history['rec_loss_diff'][-1],
                    }
                   )
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