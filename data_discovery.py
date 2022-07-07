"""
Created on Sat Feb 12 20:11:37 2022

@author: bonnyaigergo

https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from Datasets import datasets
from math import ceil


class DDiscovery:
    
    def __init__(self, dataset_name: str):
        self.df = self.import_data(dataset_name)
        self.df_melted = self.melt_df()
        self.target_mean_seq = self.df.groupby('target').mean()
        self.n_classes = datasets[dataset_name]['n_classes']
        
    def import_data(self, dataset_name: str):
    
        assert dataset_name in datasets.keys(), 'dataset_name should be one of listed in datasets dictionary'
        
        column_name = []
        n_timesteps = datasets[dataset_name]['n_timesteps']
        
        if dataset_name == 'ECG5000':
            column_name.append("target")
            for i in range(1, n_timesteps+1):
                column_name.append("ts"+str(i))
        else:        
            for i in range(1, n_timesteps+1):
                column_name.append("ts"+str(i))
            column_name.append("target")
    
        test = pd.read_csv(datasets[dataset_name]['loc'] + 'TEST.txt', 
                           sep=datasets[dataset_name]['separator'], 
                           header=None,
                           names=column_name
                           )
        train = pd.read_csv(datasets[dataset_name]['loc'] + 'TRAIN.txt', 
                            sep=datasets[dataset_name]['separator'], 
                            header=None,
                            names=column_name
                            )
        df = test.append(train, ignore_index=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        
        def change_column_order(df, col_name: str, index: int):
            cols = df.columns.tolist()
            cols.remove(col_name)
            cols.insert(index, col_name)
            return df[cols]
        
        """target column index set to 0 if it is the last one"""
        if df.columns[-1] == 'target':
            df = change_column_order(df, 'target', 0)
            
        return df
        
    def melt_df(self):
        self.df_melted = pd.melt(self.df, id_vars='target', value_vars=self.df.columns[1:], var_name="time")
        return self.df_melted

    def ts_boxplot_all_target_class(self):
        self.df_melted.boxplot(by='target', column='value')
    
    def ts_boxplot_indiv_target_class(self, target_class: int):
        self.df_melted[self.df_melted['target'] == target_class]['value'].plot(kind='box',
                                                                               title='Target Class: '+str(target_class))
    
    def ts_boxplot_sns_all_target_class(self, n_col: int):        
        sns.catplot(
            data=self.df_melted, x='time', y='value',
            col='target', kind='box', col_wrap=n_col
        )
        
    def ts_boxplot_sns_indiv_target_class(self, target_class: int):
        plt.figure(figsize=(9,5))
        sns.boxplot(x=self.df_melted[self.df_melted['target'] == target_class]['time'], 
                    y=self.df_melted[self.df_melted['target'] == target_class]['value'], 
                    palette='plasma')    
        plt.ylabel('value')
        plt.xlabel('time')
        
    def ts_hist_all_target_class(self):
        self.df_melted.hist(by='target', column='value', sharex=True)

    def ts_hist_indiv_target_class(self, target_class: int):
        self.df_melted[self.df_melted['target'] == target_class]['value'].plot(kind='hist',
                                                                               title='Target Class: ' + str(target_class))
        
    def ts_hist_timesteps_indiv_target_class(self, target_class: int):
        """check the distribution of timesteps of a given target class"""
        self.df[self.df['target'] == target_class].iloc[:,:-1].hist(figsize=(20,20), xrot=0)
        plt.show()
        
    def ts_line_central_tendency_all_target_class(self):
        self.target_mean_seq.iloc[:,:-1].T.plot.line()
        
    def ts_line_central_tendency_indiv_target_class(self, target_class: int):
        self.target_mean_seq.iloc[target_class-1,:-1].T.plot.line()

    def ts_line_central_tendency_subplots(self, n_col: int, legend=False):
        n_row = ceil(self.n_classes/n_col)
        self.target_mean_seq.iloc[:,:-1].T.plot(subplots=True, 
                                                legend=legend,
                                                layout=(n_row, n_col), 
                                                sharex=True, 
                                                sharey=True)
        
    def ts_lines_indiv_target_class_n_example(self, target_class:int, n_lines:int, 
                                              rand:bool=False, random_state:str=123):
        if rand:
            target_class_sample = self.df[self.df['target'] == target_class].sample(n=n_lines, random_state=random_state)
        else:
            target_class_sample = self.df[self.df['target'] == target_class].iloc[:n_lines+1,1:]
        
        for i in range(n_lines):
            plt.plot(target_class_sample.iloc[i,1:])

        
    # def ts_lines_indiv_target_class_n_example_create_pdf(self):
    #     for j in range(1,self.n_classes+1):
    #         with PdfPages(os.path.join(r"C:\Users\bonnyaigergo\Documents\GitHub\Deep-Temporal-Clustering\DataDiscovery\", dataset_name, "\PDF", "Linechart_Class"+str(j)+".pdf")) as export_pdf:
    #             for i in range(0,100):
    #                 plt.plot(df[df['target'] == j].iloc[i,:-1])
    #                 export_pdf.savefig()
    #                 plt.close()
    
    
if __name__ == "__main__":
    
    Data = DDiscovery('MPedestrian')
    Data.ts_boxplot_all_target_class()
    Data.ts_boxplot_indiv_target_class(target_class=2)
    Data.ts_boxplot_sns_all_target_class(n_col=5)
    Data.ts_boxplot_sns_indiv_target_class(target_class=3)
    Data.ts_hist_all_target_class()
    Data.ts_hist_indiv_target_class(target_class=3)
    Data.ts_hist_timesteps_indiv_target_class(target_class=2)
    Data.ts_line_central_tendency_all_target_class()
    Data.ts_line_central_tendency_indiv_target_class(target_class=2)
    Data.ts_line_central_tendency_subplots(n_col=3, legend=False)
    Data.ts_lines_indiv_target_class_n_example(target_class=4, n_lines=10, rand=True, random_state=None)
    


