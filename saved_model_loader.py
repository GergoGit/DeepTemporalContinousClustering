import torch
import torch.optim as optim
import itertools

import ae_models
 
def load_vae_model(trial_number):
    
    checkpoint = torch.load('runs/VAE_model'+str(trial_number)+'.pt')
    enc_arch_params = checkpoint['enc_arch_params']
    dec_arch_params = checkpoint['dec_arch_params']
    n_timesteps = checkpoint['n_timesteps']
    embedding_size = checkpoint['embedding_size']
    batch_size = checkpoint['batch_size']
    model = ae_models.VariationalAutoEncoder(ae_models.EncoderConv1D(n_timesteps, 
                                                            n_features=1, 
                                                            embedding_size=embedding_size, 
                                                            batch_size=batch_size,
                                                            enc_arch_params=enc_arch_params
                                                            ), 
                                             ae_models.DecoderConv1D(n_timesteps, 
                                                                    n_features=1, 
                                                                    embedding_size=embedding_size, 
                                                                    batch_size=batch_size,
                                                                    dec_arch_params=dec_arch_params
                                                                    )
                                      )
    
    optimizer = getattr(optim, 'Adam')(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def load_aae_model(trial_number):
    
    checkpoint = torch.load('runs/AAE_model'+str(trial_number)+'.pt')
    enc_arch_params = checkpoint['enc_arch_params']
    dec_arch_params = checkpoint['dec_arch_params']
    n_timesteps = checkpoint['n_timesteps']
    embedding_size = checkpoint['embedding_size']
    batch_size = checkpoint['batch_size']
    model = ae_models.Autoencoder(ae_models.EncoderConv1D(n_timesteps, 
                                                            n_features=1, 
                                                            embedding_size=embedding_size, 
                                                            batch_size=batch_size,
                                                            enc_arch_params=enc_arch_params
                                                            ), 
                                  ae_models.DecoderConv1D(n_timesteps, 
                                                          n_features=1, 
                                                          embedding_size=embedding_size, 
                                                          batch_size=batch_size,
                                                          dec_arch_params=dec_arch_params
                                                          )
                                  )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    discr_arch_params = checkpoint['discr_arch_params']
    discriminator = ae_models.Discriminator(embedding_size, discr_arch_params)
    discriminator.load_state_dict(checkpoint['discr_state_dict'])
    discriminator.eval()
    
    optimizer_generator = getattr(optim, 'Adam')(itertools.chain(model.encoder.parameters(), 
                                                                model.decoder.parameters(),
                                                                discriminator.parameters()
                                                                ), 
                                                                  lr=1e-4)
                                                 
    optimizer_generator.load_state_dict(checkpoint['optim_gen_state_dict'])
    optimizer_discriminator = getattr(optim, 'Adam')(discriminator.parameters(), lr=1e-3)
    optimizer_discriminator.load_state_dict(checkpoint['optim_discr_state_dict'])
    
    return model, discriminator, optimizer_generator, optimizer_discriminator

