import datetime
import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.api import SimpleExpSmoothing

from bson.binary import Binary
from utils.misc import logger
from utils.mase import mase

from ..basic_model import basic_model

class crost_v1_0_0(basic_model):
    def __init__(self):
        self.model_name = 'crost'
        self.model_version = 'v1.0.0'

    def __model_serialize__(self, model):
        return Binary(pickle.dumps(model, protocol=2))

    def __model_deserialize__(self, model):
        return pickle.loads(model)

    def __get_item_ids__(self, kitchen_id=None):
        orders = pd.read_csv('./data/orders_full.csv', index_col=0,
                    dtype = {'order_id': object,
                        'item_id': np.int32,
                        'name': object,
                        'quantity': np.int32
                        }
                )
        orders.time = pd.to_datetime(orders.time)
        ids_list = []
        for item_id in orders.item_id.unique():
            data = orders.loc[orders['item_id'] == item_id].groupby('time')[['quantity']].sum()
            daily_data = data.resample('D').sum().fillna(0)
            zeros_data = daily_data.loc[daily_data.quantity == 0]
            if len(zeros_data) < 8: #Croston is inappropriate for data that has no zero demand periods.[temp]
                continue
            else:
                ids_list.append(item_id)
        ids_list.pop(-19) #temp

        return ids_list

    def __get_data__(self, item_id, db_ai, kitchen_id=None):
        orders = pd.read_csv('./data/orders_full.csv', index_col=0,
                    dtype = {'order_id': object,
                        'item_id': np.int32,
                        'name': object,
                        'quantity': np.int32
                        }
                )
        orders.time = pd.to_datetime(orders.time)

        data = orders.loc[orders['item_id'] == item_id].groupby('time')[['quantity']].sum()
        daily_data = data.resample('D').sum().fillna(0)
        train = daily_data[:int(0.95*(len(daily_data)))]
        test = daily_data[int(0.95*(len(daily_data))):]
        non_zeros_train = train.loc[train.quantity != 0]
        data_train = data[train.index.min():train.index.max()]

        return non_zeros_train, data_train, test, data

    def __ts_intervals__(self, non_zero_ts):
        """
            non_zero_ts - time series data with datetime index and non-zero values
            returns inter demand interval time series
        """
        intervals_ts = []
        for i in range(0, len(non_zero_ts)-1):
            d = non_zero_ts.index[i+1] - non_zero_ts.index[i]
            d = d.days
            intervals_ts.append(d)
        intervals_ts.insert(0, 0)

        return intervals_ts

    def __croston__(self, nonZerosTS, intvalsData, n_days=2):
        """
           non_zeros_ts - time series data with datetime index and non-zero values
           intervals_ts - inter demand interval time series
           returns model dict and forecast
        """
        model = {} #since there are two separate ses models
        intervals_ts = self.__ts_intervals__(intvalsData)
        non_zeros_smoothing = SimpleExpSmoothing(np.asarray(nonZerosTS)).fit()
        intervals_smoothing = SimpleExpSmoothing(np.asarray(intervals_ts)).fit()
        non_zero_forecasts = non_zeros_smoothing.forecast(n_days)
        interval_forecasts = intervals_smoothing.forecast(n_days)
        forecast = non_zero_forecasts/interval_forecasts
        #forecast = np.int32(np.ceil(forecast))
        model['nonZeroSes'] = non_zeros_smoothing
        model['intervalSes'] = intervals_smoothing

        return model, forecast

    def update_model(self, db_main, db_ai, fs_ai):
        logger('NUCLEUS_MANCIO', 'REQ', 'update_model() called for: {}_{}.'.format(self.model_name, self.model_version))

        item_ids = self.__get_item_ids__()

        for item_id in item_ids:
            print('ID {}'.format(item_id))
            non_zeros_train, data_train, test, data = self.__get_data__(item_id, db_ai, kitchen_id=None)
            try:
                m, test_pred = self.__croston__(nonZerosTS=non_zeros_train, intvalsData=data_train, n_days=len(test))
                model, forecast = self.__croston__(nonZerosTS=data, intvalsData=data)
            except Exception:
                raise
            #print('In the next {} days, demand is {} units'.format(round(forecast_intervals[0], 2), round(forecast_nonzero[0], 2)))
            #This is "Important" as croston gives demand rate and not point forecasts.
            #The forecasts generated for the periods are based on the demand rate.
            residuals = test.quantity - test_pred

            metrics = {}
            #metrics['aic'] = m.aic
            #metrics['bic'] = m.bic
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

        logger('NUCLEUS_MANCIO', 'EXE', 'Update of the model: {}_{} successful!'.format(self.model_name, self.model_version))

