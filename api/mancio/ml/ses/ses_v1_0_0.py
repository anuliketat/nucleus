import os
import pickle as pkl
from math import sqrt
from bson.binary import Binary
from utils.misc import logger
from utils.mase import mase

import data
import numpy as np
import pandas as pd
import warnings
import datetime
import dateutil
from datetime import timedelta
import scipy
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.api import SimpleExpSmoothing
import warnings

from ..basic_model import basic_model
warnings.filterwarnings('ignore')

class ses_v1_0_0(basic_model):
    def __init__(self):
        self.model_name = 'ses'
        self.model_version = 'v1.0.0'

    def __get_item_ids__(self, kitchen_id = None):
        orders = pd.read_csv('./data/orders_data.csv', index_col=0, 
                dtype={'order_id':object, 
                'item_id':np.int32, 
                'name':object, 
                'quantity':np.int32})
        orders.time = pd.to_datetime(orders.time)
        ids_list = orders.item_id.unique()
        return ids_list

    def __get_data__(self, item_id, db_ai, kitchen_id = None, mode = 'daily'):
        orders = pd.read_csv('./data/orders_data.csv', index_col=0, 
            dtype={'order_id':object, 
            'item_id':np.int32, 
            'name':object, 
            'quantity':np.int32})
        orders.time = pd.to_datetime(orders.time)

        item = orders.loc[orders['item_id'] == item_id].groupby('time')[['quantity']].sum()
        if mode == 'weekly':
            data = item.resample('W').sum().fillna(0)
        elif mode == 'monthly':
            data = item.resample('M').sum().fillna(0)
        else:
            data = item.resample('D').sum().fillna(0)
        train = data[:int(0.9*(len(data)))]
        test = data[int(0.9*(len(data))):]
        return train, test, data

    def __ets__(self, ts_data, n_periods = 2): #Exponential Smoothing function
        """
            ts_data - time series data with datetime index
            n_periods - #periods to forecast
            returns model and forecasted values for the period.
        """
        model = SimpleExpSmoothing(np.asarray(ts_data)).fit()
        forecast = np.int64(np.ceil(model.forecast(n_periods)))
        return model, forecast

    def update_model(self, db_main, db_ai, fs_ai, mode = 'daily'):
        logger('NUCLEUS_MANCIO', 'REQ', 'update_model() called for: {}_{}.'.format(self.model_name, self.model_version))

        item_ids = self.__get_item_ids__()

        for item_id in item_ids:
            print('ID {}'.format(item_id))
            if mode == 'weekly':
                train, test, data = self.__get_data__(item_id, db_ai, mode ='weekly')
            elif mode == 'monthly':
                train, test, data = self.__get_data__(item_id, db_ai, mode ='monthly')
            else:
                train, test, data = self.__get_data__(item_id, db_ai)
            try:
                m, test_pred = self.__ets__(train, n_periods=len(test))
                model, forecast = self.__ets__(data)
            except ZeroDivisionError as e:
                print('aicc calculation error! ', e)
            except Exception as e:
                print('Error:', e)

            residuals = m.resid
            aic_ses = m.aic
            bic_ses = m.bic
            mse = round(mean_squared_error(test.quantity, test_pred), 3)
            mae = round(mean_absolute_error(test.quantity, test_pred), 3)
            Mase = round(mase(test.quantity, test_pred), 3)
            print('Updated for ID {}'.format(item_id))
        #Model serializing to be implemented

        #ml_model = {}
        #ml_model['modelName'] = self.model_name
        #ml_model['modelVersion'] = self.model_version
        #ml_model['modelID'] = model_id
        #ml_model['createdAt'] = datetime.datetime.utcnow()
        #db_ai.models.insert_one(ml_model)

        #_model_created_at = ml_model['createdAt']
        logger('NUCLEUS_MANCIO', 'EXE', 'Update of the model: {}_{} successful!'.format(self.model_name, self.model_version))