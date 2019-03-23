import datetime
import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.api import SimpleExpSmoothing

from bson.binary import Binary
from utils.misc import logger
from utils.mase import mase
from utils.z_score import z_score

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

    def __get_data__(self, item_id, db_ai, kitchen_id=None, mode='D'):
        orders = pd.read_csv('./data/orders_full.csv', index_col=0,
                    dtype = {'order_id': object,
                        'item_id': np.int32,
                        'name': object,
                        'quantity': np.int32
                        }
                )
        orders.time = pd.to_datetime(orders.time)

        data = orders.loc[orders['item_id']==item_id].groupby('time')[['quantity']].sum()
        daily_data = data.resample(mode).sum().fillna(0)
        train = daily_data[:int(0.95*(len(daily_data)))]
        test = daily_data[int(0.95*(len(daily_data))):]
        non_zeros_train = train.loc[train.quantity!=0]
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

    def __croston__(self, nonZerosTS, intvalsData, n_periods=2):
        """
           non_zeros_ts - time series data with datetime index and non-zero values
           intervals_ts - inter demand interval time series
           returns model dict and forecast
        """
        model = {} #since there are two separate ses models
        intervals_ts = self.__ts_intervals__(intvalsData)
        non_zeros_smoothing = SimpleExpSmoothing(np.asarray(nonZerosTS)).fit()
        intervals_smoothing = SimpleExpSmoothing(np.asarray(intervals_ts)).fit()
        non_zero_forecasts = non_zeros_smoothing.forecast(n_periods)
        interval_forecasts = intervals_smoothing.forecast(n_periods)
        forecast = non_zero_forecasts/interval_forecasts
        #forecast = np.int32(np.ceil(forecast))
        model['nonZeroSes'] = non_zeros_smoothing
        model['intervalSes'] = intervals_smoothing

        return model, forecast

    def __conf_int__(self, preds, std_err_data, alpha=0.20):
        """
            preds - forecasted time series dataframe
            st_err_data - test data and predicted test dataframe
            alpha - 0 to 1. Confidence intervals for the forecasted values. Default is 80%
            returns lower and upper conf intervals for the forecasts
        """
        z_scr = z_score(alpha)
        lower, upper = [], []
        for i in preds.forecast:
            a = i-z_scr*((mean_squared_error(std_err_data.actual, std_err_data.pred))**0.5)
            b = i+z_scr*((mean_squared_error(std_err_data.actual, std_err_data.pred))**0.5)
            lower.append(a)
            upper.append(b)

        return lower, upper

    def update_model(self, db_main, db_ai, fs_ai, mode='D'):
        logger('NUCLEUS_MANCIO', 'REQ', 'update_model() called for: {}_{}.'.format(self.model_name, self.model_version))

        item_ids = self.__get_item_ids__()

        for item_id in item_ids:
            print('Item ID:', item_id)
            non_zeros_train, data_train, test, data = self.__get_data__(item_id, db_ai, kitchen_id=None, mode=mode)
            try:
                m, test_pred = self.__croston__(nonZerosTS=non_zeros_train, intvalsData=data_train, n_periods=len(test))
                model, forecast = self.__croston__(nonZerosTS=data, intvalsData=data)
            except Exception as e:
                print(e)
            #print('In the next {} days, demand is {} units'.format(round(forecast_intervals[0], 2), round(forecast_nonzero[0], 2)))
            #This is "Important" as croston gives demand rate and not point forecasts.
            #The forecasts generated for the periods are based on the demand rate.
            test_preds = pd.DataFrame({'pred':test_pred, 'actual':test.quantity}, index=test.index)
            dates = pd.date_range(start=test.index.max()+datetime.timedelta(days=1), periods=len(forecast), freq=mode)
            forecast = pd.DataFrame(forecast, index=dates, columns=['forecast'])
            forecast['yhat_lower'], forecast['yhat_upper'] = self.__conf_int__(preds=forecast, std_err_data=test_preds)
            for col in forecast.columns:
                forecast[col] = np.where(forecast[col]<0, 0, forecast[col])
                forecast[col] = np.int64(np.ceil(forecast[col]))

            residuals = test.quantity - test_pred

            metrics = {}
            #metrics['aic'] = m.aic
            #metrics['bic'] = m.bic
            metrics['mse'] = mean_squared_error(test.quantity, test_pred)
            metrics['mae'] = mean_absolute_error(test.quantity, test_pred)
            metrics['mase'] = mase(test.quantity, test_pred)

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
            ml_model['mode'] = mode
            ml_model['createdAt'] = datetime.datetime.utcnow()
            db_ai.models.insert_one(ml_model)

        logger('NUCLEUS_MANCIO', 'EXE', 'Update of the model: {}_{} successful!'.format(self.model_name, self.model_version))

