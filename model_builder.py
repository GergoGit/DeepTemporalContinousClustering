import torch.nn as nn

def build_ae_model(n_timesteps: int, 
                    n_features: int,                    
                    enc_arch_params: dict,
                    dec_arch_params: dict,
                    embedding_size: int, 
                    batch_size: int,
                    encoder_model: nn.Module,
                    decoder_model: nn.Module,
                    ae_model: nn.Module):
        
    encoder = encoder_model(n_timesteps, 
                            n_features, 
                            embedding_size, 
                            batch_size,
                            enc_arch_params)
        
    decoder = decoder_model(n_timesteps, 
                            n_features, 
                            embedding_size, 
                            batch_size,
                            dec_arch_params)
    
    autoencoder = ae_model(encoder, decoder)
    
    return autoencoder


def build_aae_model(n_timesteps: int, 
                    n_features: int,                    
                    enc_arch_params: dict,
                    dec_arch_params: dict,
                    discr_arch_params: dict,
                    embedding_size: int, 
                    batch_size: int,
                    encoder_model: nn.Module,
                    decoder_model: nn.Module,
                    ae_model: nn.Module,
                    discriminator_model: nn.Module):
        
    encoder = encoder_model(n_timesteps, 
                            n_features, 
                            embedding_size, 
                            batch_size,
                            enc_arch_params)
        
    decoder = decoder_model(n_timesteps, 
                            n_features, 
                            embedding_size, 
                            batch_size,
                            dec_arch_params)
    
    autoencoder = ae_model(encoder, decoder)
    
    discriminator = discriminator_model(embedding_size,
                                        discr_arch_params)
    
    return autoencoder, discriminator

def build_gan_model(n_timesteps: int, 
                    n_features: int,  
                    embedding_size: int, 
                    batch_size: int,
                    enc_arch_params: dict,
                    dec_arch_params: dict,
                    encoder_model: nn.Module,
                    decoder_model: nn.Module,
                    ae_model: nn.Module,
                    discr_arch_params: dict,
                    discriminator_model: nn.Module,
                    gen_arch_params: dict,
                    generator_model: nn.Module):
        
    encoder = encoder_model(n_timesteps, 
                            n_features, 
                            embedding_size, 
                            batch_size,
                            enc_arch_params)
        
    decoder = decoder_model(n_timesteps, 
                            n_features, 
                            embedding_size, 
                            batch_size,
                            dec_arch_params)
    
    autoencoder = ae_model(encoder, decoder)
    
    generator = generator_model(n_timesteps, 
                                n_features,
                                batch_size,
                                gen_arch_params)
    
    discriminator = discriminator_model(embedding_size, 
                                        discr_arch_params)
    
    # TODO: test different generators with large and small input size
    
    return autoencoder, generator, discriminator