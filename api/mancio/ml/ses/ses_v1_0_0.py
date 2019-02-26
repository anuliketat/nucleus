import os
import pickle as pkl
from math import sqrt
from bson.binary import Binary
from .mase import mase

import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.api import SimpleExpSmoothing

from ..basic_model import basic_model

warnings.filterwarnings('ignore')

class ses_v1_0_0(basic_model):
	def __init__(self):
		self.model_name = 'ses'
		self.model_version = 'v1.0.0'

	def __get_data__(self, kitchen_id = None, item_id, mode = 'daily', db_ai):
		orders = pd.read_csv('orders_data.csv', index_col=0, 
			dtype={'order_id':object, 
			'item_id':np.int32, 
			'name':object, 
			'quantity':np.int32})
		orders.time = pd.to_datetime(orders.time)
		
		item = orders.loc[orders['item_id'] == id].groupby('time')[['quantity']].sum()
		if mode == 'weekly':
			data = item.resample('W').sum().fillna(0)
		elif mode == 'monthly':
			data = item.resample('M').sum().fillna(0)  
		else:
			data = item.resample('D').sum().fillna(0) 
		train = data[:int(0.9*(len(data)))]
		test = data[int(0.9*(len(data))):]
		return train, test, data

	def __ets__(self, ts_data, n_periods = 1): #Exponential Smoothing function
		"""
			ts_data - time series data with datetime index
			n_periods - #periods to forecast
		"""
		model = SimpleExpSmoothing(np.asarray(ts_data)).fit()
		forecast = np.int32(np.ceil(fit_test.forecast(n_periods)))
		return model, forecast

	def update_model(self, item_id, db_main, db_ai, fs_ai):
		logger('NUCLEUS_MANCIO', 'REQ', 'update_model() called for: {}_{}.'.format(self.model_name, self.model_version))
		
        modes = ['daily', 'weekly', 'monthly']
		Item = {}
        for mode in modes:
            train, test, data = self.__get_data__(db_ai, item_id, mode = mode)
			try:
				m, test_pred = self.__ets__(train)
				model, forecast = self.__ets__(data)
			except ZeroDivisionError as e:
				print('aicc calculation error!', e)
				continue
			except Exception as e:
				print(e)
				continue
			Item['{}_{}'.format(item_id, mode)] = forecast # dictionary of forecasts by mode. Ex: {'80_daily': array([4]),
															#										'80_weekly': array([18]),
															#										'80_monthly': array([172])}

		residuals = m.resid
		aic_ses = m.aic
		bic_ses = m.bic 
		metrics = pd.DataFrame({'aic': round(aic_ses, 3), 'bic': round(bic_ses, 3),  
	                           'mse': round(mean_squared_error(test.quantity, test_pred), 3), 
	                               'mae': round(mean_absolute_error(test.quantity, test_pred), 3),
	                               'mase': round(mase(test.quantity, test_pred), 3)}, 
	                          index = orders.loc[orders.item_id == id].item_id.unique(), dtype = np.float32)

		#Model serializing to be implemented

		#ml_model = {}
		#ml_model['modelName'] = self.model_name
		#ml_model['modelVersion'] = self.model_version
		#ml_model['modelID'] = model_id
		#ml_model['createdAt'] = datetime.datetime.utcnow()
		#db_ai.models.insert_one(ml_model)

		#_model_created_at = ml_model['createdAt']
		logger('NUCLEUS_MANCIO', 'EXE', 'Update of the model: {}_{} successful!'.format(self.model_name, self.model_version))
