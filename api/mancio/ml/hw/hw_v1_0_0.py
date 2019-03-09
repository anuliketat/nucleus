import pickle
from bson.binary import Binary
from utils.misc import logger
from utils.mase import mase

import numpy as np
import pandas as pd
import datetime

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.api import ExponentialSmoothing

from ..basic_model import basic_model

class hw_v1_0_0(basic_model):
    def __init__(self):
        self.model_name = 'hw'
        self.model_version = 'v1.0.0'

    def __model_serialize__(self, model):
        return Binary(pickle.dumps(model, protocol=2))

    def __model_deserialize__(self, model):
        return pickle.loads(model)

    def __get_item_ids__(self, kitchen_id=None):
        orders = pd.read_csv('./data/orders_data.csv', index_col=0,
                    dtype = {'order_id': object,
                        'item_id': np.int32,
                        'name': object,
                        'quantity': np.int32
                        }
                )
        orders.time = pd.to_datetime(orders.time)
        ids_list = orders.item_id.unique()
        return ids_list

    def __get_data__(self, item_id, db_ai, kitchen_id=None, mode='daily'):
        orders = pd.read_csv('./data/orders_data.csv', index_col=0,
                    dtype = {'order_id': object,
                        'item_id': np.int32,
                        'name': object,
                        'quantity': np.int32
                        }
                )
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

    def __hw__(self, ts_data, n_periods = 2, mode = 'daily'):
        """
            ts_data - time series data with datetime index
            n_periods - #periods to forecast
            returns model and forecasted values for the period.
        """
        if mode == 'weekly':
            model = ExponentialSmoothing(np.asarray(ts_data['quantity']), seasonal_periods = 4, trend = 'add', seasonal = 'add').fit()
        elif mode == 'monthly':
            model = ExponentialSmoothing(np.asarray(ts_data['quantity']), seasonal_periods = 1, trend = 'add', seasonal = 'add').fit()
        else:
            model = ExponentialSmoothing(np.asarray(ts_data['quantity']), seasonal_periods = 30, trend = 'add', seasonal = 'add').fit()
        
        forecast = np.int64(np.ceil(model.forecast(steps = n_periods)))
        forecast = np.where(forecast < 0, 0, forecast)
        return model, forecast

    def update_model(self, db_main, db_ai, fs_ai, mode = 'daily'):
        logger('NUCLEUS_MANCIO', 'REQ', 'update_model() called for: {}_{}.'.format(self.model_name, self.model_version))
        
        item_ids = self.__get_item_ids__()

        for item_id in item_ids:
            print('Item ID: {}'.format(item_id))
            try:
                if mode == 'weekly':
                    train, test, data = self.__get_data__(item_id, db_ai, mode ='weekly')
                    m, test_pred = self.__hw__(train, n_periods=len(test), mode = 'weekly')
                    model, forecast = self.__hw__(data, mode = 'weekly')
                elif mode == 'monthly':
                    train, test, data = self.__get_data__(item_id, db_ai, mode ='monthly')
                    m, test_pred = self.__hw__(train, n_periods=len(test), mode = 'monthly')
                    model, forecast = self.__hw__(data, mode = 'monthly')
                else:
                    train, test, data = self.__get_data__(item_id, db_ai)
                    m, test_pred = self.__hw__(train, n_periods=len(test))
                    model, forecast = self.__hw__(data)   
            except Exception:
                raise

            residuals = test.quantity - test_pred

            metrics = {}
            metrics['aic'] = m.aic
            metrics['bic'] = m.bic
            metrics['mse'] = round(mean_squared_error(test.quantity, test_pred), 3)
            metrics['mae'] = round(mean_absolute_error(test.quantity, test_pred), 3)
            metrics['mase'] = round(mase(test.quantity, test_pred), 3)

            _model = {}
            _model['data'] = data
            _model['forecast'] = forecast
            _model['residuals'] = residuals
            _model['model'] = model
            model_id = fs_ai.put(self.__model_serialize__(_model))

            ml_model = {}
            ml_model['modelName'] = self.model_name
            ml_model['modelVersion'] = self.model_version
            ml_model['modelID'] = model_id
            ml_model['metrics'] = metrics
            ml_model['createdAt'] = datetime.datetime.utcnow()
            db_ai.models.insert_one(ml_model)
            print('Updated {} for ID {}'.format(self.model_name, item_id))

        logger('NUCLEUS_MANCIO', 'EXE', 'Update of the model: {}_{} successful!'.format(self.model_name, self.model_version))