import datetime
import pickle

import numpy as np
import pandas as pd
from bson.binary import Binary
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from api.exceptions import NoModel
from utils.mase import mase
from utils.misc import get_traceback, logger
from utils.z_score import z_score
from utils.io import model_serialize

from ..basic_model.basic_model import basic_model

class crost_v1_0_0(basic_model):
    def __init__(self):
        self.model_name = 'crost'
        self.model_version = 'v1.0.0'

    def __get_data__(self, item_data_id, db_main, kitchen_id, mode='D'):
        items, item_data_ids, quantity, time, order_ids, kitchen_ids = [], [], [], [], [], []
        kit_id = {}
        for data in db_main.ordered_items.find():
            items.append(data.get('name'))
            item_data_ids.append(data.get('item_data_id'))
            quantity.append(data.get('quantity'))
            time.append(data.get('timestamp'))
            order_ids.append(data.get('order_id'))
            kitchen_ids.append(data.get('kitchen_id'))

        all_orders = pd.DataFrame(
            {
                "order_id": order_ids,
                "kitchen_id": kitchen_ids,
                "item_data_id": item_data_ids,
                "name": items,
                "quantity": quantity,
                "time": time,
            }
        )
        all_orders = all_orders[all_orders.item_data_id != None]
        all_orders.time = pd.to_datetime(all_orders.time, dayfirst=True, format = '%Y-%m-%d %H:%M').dt.date
        all_orders.time = pd.to_datetime(all_orders.time)

        for k in all_orders.kitchen_id.unique():
            kit_id[k] = all_orders.loc[all_orders.kitchen_id==k]
        orders = kit_id[kitchen_id]
        data = orders.loc[orders['item_data_id']==item_data_id].groupby('time')[['quantity']].sum()
        daily_data = data.resample(mode).sum().fillna(0)
        train = daily_data[:int(0.9*(len(daily_data)))]
        test = daily_data[int(0.9*(len(daily_data))):]
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
        item_data_ids, kitchen_ids = [], []
        for d in db_main.ordered_items.find():
            item_data_ids.append(d.get('item_data_id'))
            kitchen_ids.append(d.get('kitchen_id'))
        item_data_ids = [i for i in item_data_ids if i]

        for kitchen_id in set(kitchen_ids):
            for item_data_id in set(item_data_ids):
                print('\nItem ID:', item_data_id)
                non_zeros_train, data_train, test, data = self.__get_data__(item_data_id, db_main, kitchen_id=kitchen_id, mode=mode)
                try:
                    m, test_pred = self.__croston__(nonZerosTS=non_zeros_train, intvalsData=data_train, n_periods=len(test))
                    model, forecast = self.__croston__(nonZerosTS=data, intvalsData=data)
                except ValueError as e:
                    logger('NUCLEUS_MANCIO', 'ERR', get_traceback(e))
                    logger('NUCLEUS_MANCIO', 'ERR', 'Data has only 1 record')
                    logger('NUCLEUS_MANCIO', 'ERR', 'Error in update_model() for {}_{} and item_id={} with mode={}.'.format(self.model_name, self.model_version, item_data_id, mode))
                    continue
                except NotImplementedError as e:
                    logger('NUCLEUS_MANCIO', 'ERR', get_traceback(e))
                    logger('NUCLEUS_MANCIO', 'ERR', 'Not enough data')
                    logger('NUCLEUS_MANCIO', 'ERR', 'Error in update_model() for {}_{} and item_id={} with mode={}.'.format(self.model_name, self.model_version, item_data_id, mode))
                    continue
                except Exception as e:
                    logger('NUCLEUS_MANCIO', 'ERR', get_traceback(e))
                    logger('NUCLEUS_MANCIO', 'ERR', 'Error in update_model() for {}_{} and item_id={} with mode={}.'.format(self.model_name, self.model_version, item_data_id, mode))
                    continue

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
                _model['kitchen'] = kitchen_id
                _model['itemDataID'] = item_data_id
                _model['data'] = data
                _model['forecast'] = forecast
                _model['residuals'] = residuals
                _model['model'] = model
                model_id = fs_ai.put(model_serialize(_model))

                ml_model = {}
                ml_model['modelName'] = self.model_name
                ml_model['modelVersion'] = self.model_version
                ml_model['modelID'] = model_id
                ml_model['kitchenID'] = kitchen_id
                ml_model['itemDataID'] = item_data_id
                ml_model['metrics'] = metrics
                ml_model['mode'] = mode
                ml_model['createdAt'] = datetime.datetime.utcnow()
                db_ai.mancioModels.update_one({'kitchenID': kitchen_id, 'itemDataID': item_data_id, 'modelName': self.model_name, 'modelVersion': self.model_version}, {'$set': ml_model}, upsert=True)
