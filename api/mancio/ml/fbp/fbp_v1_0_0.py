import pandas as pd
import numpy as np
import datetime

from bson.binary import Binary
from utils.misc import logger
from utils.mase import mase

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric

import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ..basic_model import basic_model

class fbp_v1_0_0(basic_model):
    def __init__(self):
        self.model_name = 'fbp'
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
        for df in [train, test, data]:
            df.index.name = 'ds'
            df.columns = ['y']
            df = df.reset_index(inplace=True)
        return train, test, data

    def __df_holidays__(self):
        holidays = pd.read_csv('./data/fbp_holidays.csv', index_col=0, 
                            dtype = {'holiday': object, 'ds':object})
        holidays.ds = pd.to_datetime(holidays.ds)
        return holidays

    def __fbp__(self, ts_data, n_periods = 2):
        """
            ts_data - time series data with time column as 'ds' and value as 'y'
        """
        holidays = self.__df_holidays__()
        model = Prophet(holidays = holidays, daily_seasonality = False, weekly_seasonality = True, yearly_seasonality = False) #conf.int = 80%
        model.fit(ts_data)
        future = model.make_future_dataframe(periods = n_periods, include_history = False)
        forecast = model.predict(future)
        forecast = forecast[['ds', 'yhat_lower', 'yhat_upper', 'yhat']]
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                forecast[col] = np.where(forecast[col] < 0, 0, forecast[col])
                forecast[col] = np.int64(np.ceil((forecast[col])))
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
                m, test_pred = self.__fbp__(train, n_periods = len(test))
                model, forecast = self.__fbp__(data)
            except Exception:
                raise

            residuals = test.y - test_pred.yhat
            metrics = {}
            #metrics['aic'] = m.aic
            #metrics['bic'] = m.bic
            metrics['mse'] = round(mean_squared_error(test.y, test_pred.yhat), 3)
            metrics['mae'] = round(mean_absolute_error(test.y, test_pred.yhat), 3)
            metrics['mase'] = round(mase(test.y, test_pred.yhat), 3)

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

        logger('NUCLEUS_MANCIO', 'EXE', 'Update of the model: {}_{} successful!'.format(self.model_name, self.model_version))