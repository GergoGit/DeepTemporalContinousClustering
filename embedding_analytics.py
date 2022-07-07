import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.manifold import TSNE
import umap.umap_ as umap
import trimap
import pacmap
import seaborn as sns
import plotly.express as px
import plotly.io as pio
# pio.renderers.default = "browser"
pio.renderers.default = "svg"
from plotly.offline import plot
import torch
import torch.nn as nn


##################
# Plotting functions
##################

def Scatter_2D(embedding, true_labels, title):
    
    df = pd.DataFrame(embedding)
    col_num = len(pd.unique(true_labels))

    ax = sns.scatterplot(
        x=0, y=1,
        hue=true_labels,
        palette=sns.color_palette("hls", col_num),
        data=df,
        legend="full",
        alpha=0.3
    )
    ax.set_title(title)
    ax.set_xlabel('Dim1')
    ax.set_ylabel('Dim2')

def Scatter2D(embedding, true_labels, title):    
    fig = px.scatter(
        embedding, x=0, y=1, title=title,
        color=true_labels, labels={'color': 'level'}
    )
    plot(fig)
    
def Scatter_3D(embedding, true_labels, title):
    fig = px.scatter_3d(
    embedding, x=0, y=1, z=2, title=title,
    color=true_labels, labels={'color': 'level'}
    )
    plot(fig)
    

if __name__ == "__main__":
    
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

    import datasets
    from utils import embedding_histogram
    from datasets import CustomDataset
    import saved_model_loader as sml

    
    DATASET = 'Crop'
    MODEL_TYPE = 'VAE'
    TRIAL_NUMBER = 0
    SAMPLE_SIZE = 1000
    
    output_dir = 'runs/' #+DATASET+'/'
    
    checkpoint = torch.load(output_dir+MODEL_TYPE+'_model'+str(TRIAL_NUMBER)+'.pt')
    # batch_size = checkpoint['batch_size']
    
    ######################
    # Data 
    ######################
    
    X, y, _, _ = datasets.prepare_inputdata(dataset_name=DATASET)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=123)
    
    train_set = CustomDataset(X_train, train=True)
    train_dataloader = DataLoader(train_set, batch_size=SAMPLE_SIZE, shuffle=False)
    
    true_labels = pd.Series(y_train[:SAMPLE_SIZE], name = "target")
    
    ######################
    # Load Model 
    ######################
    
    # model = sml.load_vae_model(0)
    model, _, _, _ = sml.load_aae_model(0)
    model.encoder.batch_size = SAMPLE_SIZE
    model.decoder.batch_size = SAMPLE_SIZE
    
    sample = next(iter(train_dataloader))
    embedding = model.encoding(sample).to('cpu')
    embedding = embedding.detach().numpy()
    
    ######################
    # Check embedding histogram
    ######################    
    
    n_rows, n_cols = 4, 4
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, 
                             sharex=True, sharey=True,
                             # figsize=(15, 15)
                             )
    
    i = 0
    for row in range(len(axes)):
        for col in range(len(axes[0])):
            axes[row][col].hist(embedding[:, i])
            i += 1
    plt.show()
    
    ######################
    # Rec loss histogram
    ######################    
    
    predicted = model(sample)
    differences = (predicted - sample)**2
    losses = torch.mean(input=differences, dim=1)
    
    plt.hist(losses.detach().numpy(), density=True, bins=30)  # density=False would make counts
    plt.ylabel('Prob')
    plt.xlabel('Sample Loss')
    
    ######################
    # Generated vs Original
    ######################   
    
    seq_number = 9
    
    plt.plot(sample[seq_number].detach().numpy())
    plt.plot(predicted[seq_number].detach().numpy())
    plt.ylabel('y')
    plt.xlabel('time')
    plt.title('Sample check')
    plt.legend(['real','generated'])
    plt.show()
    
    
    ######################
    # Dim Reduction
    ######################
    
    reducer = TSNE(n_components = 2, 
                    perplexity = 10.0, 
                    n_iter = 500, 
                    verbose = 1)
    reducer = umap.UMAP(n_components=2)
    reducer = trimap.TRIMAP(n_dims=2)
    reducer = pacmap.PaCMAP(n_components=2)
        
    projections_2d = reducer.fit_transform(embedding)
    
    reducer = TSNE(n_components = 3, 
                    perplexity = 10.0, 
                    n_iter = 500, 
                    verbose = 1)
    reducer = umap.UMAP(n_components=3)
    reducer = trimap.TRIMAP(n_dims=3)
    reducer = pacmap.PaCMAP(n_components=3)
    
    projections_3d = reducer.fit_transform(embedding)
    
    ######################
    # Plot
    ######################
    
    Scatter_2D(embedding=projections_2d, true_labels=true_labels, title="UMAP 2D")
    Scatter2D(embedding=projections_2d, true_labels=true_labels, title="UMAP 2D")
    
    Scatter_3D(embedding=projections_3d, true_labels=true_labels, title='T-SNE 3D')
