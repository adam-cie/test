from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

class MSD_Transformer(TransformerMixin):

    #---------------------------------------------------------
        #INTERNAL FUNCTIONS
    #---------------------------------------------------------

    def normalizeData(self, data):
        ###TO DO
        data_ = data.copy()
        data_t = data_.transpose()
        if self.expert_range is None:
            for i in range(self.x):
                col_min, col_max = min(data_t[i]), max(data_t[i])
                for j in range(self.y):
                    data_[j][i] = (data[j][i] - col_min)/(col_max - col_min)
        else:
            for i in range(self.x):
                for j in range(self.y):
                    data_[j][i] = (data[j][i] - self.expert_range[i][0])/(self.expert_range[i][1] - self.expert_range[i][0])
        data_ = data_.transpose()
        for i in range(self.x):
            if self.objectives[i] == 'min':
                data_[i] = np.ones(self.y) - data_[i]
        data_ = data_.transpose()
        return data_

    def normalizeWeights(self, weights):
        ###TO DO
        weights = np.array([float(i)/max(weights) for i in weights])
        return weights

    def calulateMean(self):
        ###TO DO
        mean = np.zeros(self.y)
        for i in range(self.y):
            mean[i] = np.average(self.data[i])
        return mean

    def calculateSD(self):
        ###TO DO
        sd = np.zeros(self.y)
        for i in range(self.y):
            sd[i] = np.std(self.data[i])
        return sd

    def topsis(self):
        ###TO DO
        topsis_val = np.zeros(self.y)
        if self.agg_fn == 'I':
            for i in range(self.y):
                topsis_val[i] = 1 - np.sqrt((1-self.mean_col[i])*(1-self.mean_col[i])+(self.sd_col[i]*self.sd_col[i]))
        elif self.agg_fn == 'A':
            for i in range(self.y):
                topsis_val[i] = np.sqrt(self.mean_col[i]*self.mean_col[i]+(self.sd_col[i]*self.sd_col[i]))
        else:
            for i in range(self.y):
                topsis_val[i] = (np.sqrt(self.mean_col[i]*self.mean_col[i]+(self.sd_col[i]*self.sd_col[i])))/(((1 - np.sqrt((1-self.mean_col[i])*(1-self.mean_col[i])+(self.sd_col[i]*self.sd_col[i])))-1)*(-1) + (np.sqrt(self.mean_col[i]*self.mean_col[i]+(self.sd_col[i]*self.sd_col[i]))))
        return topsis_val

    def ranking(self):
        ###TO DO
        arranged = self.alternatives.copy()
        val = self.topsis_val.argsort()
        arranged = arranged[val[::-1]]
        return arranged
        #arranged = []
        #arranged = self.data.copy()
        #arranged['R'] = self.data.topsis_val['R']
        #arranged = arranged.sort('R', ascending = False)
        #return arranged[:-1]

    #---------------------------------------------------------
        #EXTERNAL FUNCTIONS
    #---------------------------------------------------------
    
    def __init__(self, data, criteria = None, alternatives = None, weights = None, objectives = None, expert_range = None, agg_fn = 'I'):
        
        # data
        self.data = data

        #store information about number of rows and number of columns (excluding headers)
        self.x = self.data.shape[1] # number of columns
        self.y = self.data.shape[0] # number of rows

        # [optional] column names: default 0,1, ... x
        self.criteria = (criteria if criteria is not None else np.arange(self.x))

        # [optional] row names: default 0,1, ... y
        self.alternatives = (alternatives if alternatives is not None else np.arange(self.y))

        # [optional] criteria weights: default 1, 1, ... 1
        self.weights = (weights if weights is not None else np.ones(self.y))

        # [optional] which criteria should be min, which max: deault max, max, ... max
        self.objectives = (objectives if objectives is not None else np.repeat( 'max', self.y))

        # [optional] expert range: default None
        self.expert_range = expert_range 

        # [optional] I R or A: default I
        self.agg_fn = agg_fn                

        # store values of caluclating ranked alternatives, mean, sd and topsis value
        self.mean_col = []
        self.sd_col = []
        self.topsis_val = []
        self.ranked_alternatives = []


    def fit(self, target):
        return self

    def transform(self):

        # create a copy of data to avoid changes to original dataset
        data_ = self.data.copy()
        weights_ = self.weights.copy()

        #normalize data
        data_ = self.normalizeData(data_)
        self.data = data_

        #normalize weights
        weights_ = self.normalizeWeights(weights_)
        self.weights = weights_

        #MSD transformation
        self.mean_col = self.calulateMean()
        self.sd_col = self.calculateSD()
        self.topsis_val = self.topsis()

        #ranking
        self.ranked_alternatives = self.ranking()

        return data_ #probably needs change to pandas

    def inverse_transform(self, target):
        #TO DO
        target_ = target.copy()

        return target_

    def plot(self, target):
        #TO DO
        #plot_data = self.alternatives.copy()
        #plot_data['Mean'] = self.mean_col
        #plot_data['SD'] = self.sd_col
        
        #plot_data.plot(plot_data['Mean'], plot_data['SD'], o)
        #plot_data.xlabel("Mean")
        #plot_data.ylabel("SD")
        #plt.show()

        return None
    

