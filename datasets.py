# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 20:00:26 2021

@author: bonnyaigergo

http://www.timeseriesclassification.com/description.php?Dataset=MelbournePedestrian
http://www.timeseriesclassification.com/description.php?Dataset=Crop
http://timeseriesclassification.com/description.php?Dataset=ECG5000

DataLoader
https://visualstudiomagazine.com/articles/2020/09/10/pytorch-dataloader.aspx
https://aigeekprogrammer.com/data-preparation-with-dataset-and-dataloader-in-pytorch/
https://www.youtube.com/watch?v=Sj-gIb0QiRM

"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

dataset_dict = {'Crop': {'n_observations': 24_000,
                         'n_timesteps': 46,
                         'n_features': 1,
                         'n_classes': 24,
                         'loc': r'C:\Users\bonnyaigergo\Documents\GitHub\Deep-Temporal-Clustering\Data\Crop\Crop_',
                         'separator': ',',
                         'target_col_idx': -1,
                         'class_dict': {
                                        1: "corn",
                                        2: "wheat",
                                        3: "dense building",
                                        4: "built indu",
                                        5: "diffuse building",
                                        6: "temporary meadow",
                                        7: "hardwood",
                                        8: "wasteland",
                                        9: "jachere",
                                        10: "soy",
                                        11: "water",
                                        12: "pre",
                                        13: "softwood",
                                        14: "sunflower",
                                        15: "sorghum",
                                        16: "eucalyptus",
                                        17: "rapeseed",
                                        18: "but drilling",
                                        19: "barley",
                                        20: "peas",
                                        21: "poplars",
                                        22: "mineral surface",
                                        23: "gravel",
                                        24: "lake"
                                        }
                         },
                'MPedestrian': {'n_observations': 3_633,
                                'n_timesteps': 24,
                                'n_features': 1,
                                'n_classes': 10,
                                'loc': r'C:\Users\bonnyaigergo\Documents\GitHub\Deep-Temporal-Clustering\Data\MelbournePedestrian\Pedestrian_',
                                'separator': ',',
                                'target_col_idx': -1,
                                'class_dict': {
                                                1: "Bourke Street Mall (North)",
                                                2: "Southern Cross Station",
                                                3: "New Quay",
                                                4: "Flinders St Station Underpass",
                                                5: "QV Market-Elizabeth (West)",
                                                6: "Convention/Exhibition Centre",
                                                7: "Chinatown-Swanston St (North)",
                                                8: "Webb Bridge",
                                                9: "Tin Alley-Swanston St (West)",
                                                10: "Southbank"
                                                }
                                },
                'ECG5000': {'n_observations': 5_000,
                            'n_timesteps': 140,
                            'n_features': 1,
                            'n_classes': 5,
                            'loc': r'C:\Users\bonnyaigergo\Documents\GitHub\Deep-Temporal-Clustering\Data\ECG5000\ECG5000_',
                            'separator': '\s+',
                            'target_col_idx': 0
                            }
                }


def ts_scaler(X: np.array) -> list:
    """
    Parameters
    ----------
    X : np.array
        Time series observations without target variable.

    Returns
    -------
    X_scaled : np.array
    X_min : float
    X_max : float

    """
    X_min = np.min(X)
    X_max = np.max(X)
    X_scaled = np.empty(shape=X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_scaled[i,j] = (X[i,j] - X_min)/(X_max - X_min)
            
    return X_scaled, X_min, X_max

# def create_noisy_ts(X: np.array, noisy_prop: float, std_multiplier: float=0.5):
#     n_noisy_ts = np.round(X.shape[0]*noisy_prop)
#     X_noise = np.empty(shape=(n_noisy_ts, X.shape[1]))
    
#     mu = 0
#     sigma = np.std(X)*std_multiplier
    
#     np.random.seed(123)
#     for i in range(X.shape[0]):
#         for j in range(X.shape[1]):
#             X_noise[i,j] = X[i,j] + np.random.normal(loc=mu, scale=sigma, size=1)
#     X_added_noise = np.concatenate((X, X_noise), axis=0)
#     target_noise = np.concatenate((np.zeros(shape=X.shape[0]), np.ones(shape=X_noise.shape[0])), axis=None)
#     return X_added_noise, target_noise
    

def prepare_inputdata(dataset_name: str, 
                      add_noise: bool=False,
                      noisy_prop: float=0.2) -> list:
    
    assert dataset_name in dataset_dict.keys(), 'dataset_name should be one of listed in dataset_dict dictionary'
        
    test = pd.read_csv(dataset_dict[dataset_name]['loc'] + 'TEST.txt', 
                      sep=dataset_dict[dataset_name]['separator'], header=None
                      )
    train = pd.read_csv(dataset_dict[dataset_name]['loc'] + 'TRAIN.txt', 
                      sep=dataset_dict[dataset_name]['separator'], header=None
                      )
    
    df = test.append(train, ignore_index=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    
    X = df.drop(df.columns[[dataset_dict[dataset_name]['target_col_idx']]], axis=1).to_numpy()
    y = df.iloc[:,dataset_dict[dataset_name]['target_col_idx']].to_numpy()

    # if add_noise:
    #     X, target_noise = create_noisy_ts(X, noisy_prop, std_multiplier=0.5)
    
    X_scaled, X_min, X_max = ts_scaler(X)
    
    # Turn it into float32 type and tensor format
    X_scaled = torch.from_numpy(X_scaled).type(torch.FloatTensor)
    y = torch.from_numpy(y).type(torch.FloatTensor)
            
    return X_scaled, y, X_min, X_max


class CustomDataset(Dataset):
    """Custom dataset loader"""

    def __init__(self, X, train=True):
        self.train = train
        if self.train:
            self.X_train = X
        else:
            self.X_val = X

    def __len__(self):
        if self.train:
            return self.X_train.shape[0]
        else:
            return self.X_val.shape[0]

    def __getitem__(self, item):
        if self.train:
            return self.X_train[item]
        else:
            return self.X_val[item]


if __name__ == "__main__":
    
    # X, y, X_min, X_max = prepare_inputdata(dataset_name='Crop')
    # X, y, X_min, X_max = prepare_inputdata(dataset_name='ECG5000')
    X, y, X_min, X_max = prepare_inputdata(dataset_name='MPedestrian')
    
    #######################################
    # Check the distribution of classes
    #######################################
    
    label_dict = {'target': y}
    Y = pd.DataFrame(label_dict)
    D = Y.groupby(['target'])['target'].count()
    
    import matplotlib.pyplot as plt
    y_pos = np.arange(len(D))
    plt.bar(y_pos, D)
    plt.xticks(y_pos, D.index)
    plt.show()
    
    

