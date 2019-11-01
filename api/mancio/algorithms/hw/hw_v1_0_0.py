import datetime
import pickle

import numpy as np
import pandas as pd
from bson.binary import Binary
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from api.exceptions import NoModel
from utils.mase import mase
from utils.misc import get_traceback, logger
from utils.z_score import z_score
from utils.io import model_serialize

from ..basic_model.basic_model import basic_model

class hw_v1_0_0(basic_model):
    def __init__(self):
        self.model_name = 'hw'
        self.model_version = 'v1.0.0'

    def __get_data__(self, item_data_id, db_main, kitchen_id, mode='D'):
        # using mongodb aggregation pipeline.
        # This requires timestamp to be in date format and in same collection as ordered_items data
        pipeline = [
                    {'$match': {'kitchen_id':kitchen_id, 'item_data_id': item_data_id}},
                    {'$project':{
                                  'date': { '$dateToString': { 'format': "%Y-%m-%d", 'date': "$order_time" } },
                                  'time': { '$dateToString': { 'format': "%H:%M:%S", 'date': "$order_time" } },
                                  'item_data_id':1,
                                  'quantity':1
                                } },
                    {'$group': {'_id': '$date', 'quant': {'$sum': '$quantity'}}},
                    {'$sort': {'_id':1}}
                ]
        quantity, time = [], []
        for data in db_main.ordered_items.aggregate(pipeline):
            time.append(data.get('_id'))
            quantity.append(data.get('quant'))

        item = pd.DataFrame({'quantity':quantity, 'time':time})
        item.dropna(axis=0, inplace=True)
        item.time = pd.to_datetime(item.time, dayfirst=True, format='%Y-%m-%d %H:%M').dt.date #sparse date from datetime
        item.time = pd.to_datetime(item.time)
        item.set_index('time', inplace=True)
        print('item records: ', item.shape[0])
        data = item.resample(mode).sum().fillna(0)
        train = data[:int(0.9*(len(data)))]
        test = data[int(0.9*(len(data))):]

        return train, test, data

    def __seasonal_periods__(self, ts_data, mode='D'):
        """
            ts_data - time series data
        """
        if mode == 'W':
            if len(ts_data) <= 3:
                s_periods, s_type = None, None
            else:
                s_periods = 2
                s_type = 'add'
        elif mode == 'M':
            if len(ts_data) < 2:
                s_periods, s_type = None, None
            else:
                s_periods = 2
                s_type = 'add'
        else:
            if len(ts_data) <= 14:
                s_periods, s_type = None, None
            else:
                s_periods = 14
                s_type = 'add'

        return s_periods, s_type

    def __hw__(self, ts_data, s_periods, s_type, n_periods, mode='D'):
        """
            ts_data - time series data with datetime index and 'quantity' column
            n_periods - #periods to forecast
            returns model and forecasted values for the period.
        """
        model = ExponentialSmoothing(np.asarray(ts_data['quantity']), seasonal_periods=s_periods, trend='add', seasonal=s_type).fit()
        forecast = model.forecast(steps = n_periods)

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

    def update_model(self, db_main, db_ai, fs_ai, n_periods, mode='D'):
        """
            Updates and saves the model with new data in the mancioModels collection.
        """
        item_data_ids, kitchen_ids = [], []
        for data in db_main.ordered_items.find():
            item_data_ids.append(data.get('item_data_id'))
            kitchen_ids.append(data.get('kitchen_id'))
        orders = pd.DataFrame({'kit_id':kitchen_ids, 'item_data_id':item_data_ids}) # for looping on items per kitchen since we have ids repeating in kitchens
        orders.dropna(axis=0, inplace=True)
        #item_data_ids = [103]
        data_less_items = {}
        no_data_items = [] #a blacklist for items with no data
        for kitchen_id in sorted(set(kitchen_ids)):
            items = sorted(orders[orders.kit_id==kitchen_id].item_data_id.unique())
            items = [int(i) for i in items if i]
            print('-'*45, 'Kitchen ID: {}'.format(kitchen_id), '-'*45)
            for item_data_id in items:
                print('\nKitchen: {} | Item Data ID: {} | Mode: {} | Model: {}'.format(kitchen_id, item_data_id, mode, self.model_name))
                train, test, data = self.__get_data__(item_data_id, db_main, kitchen_id, mode=mode)
                if len(data) <= 2:
                    no_data_items.append(tuple((kitchen_id, item_data_id)))
                    kit_items = [no_data_items[i][1] for i in range(len(no_data_items)) if no_data_items[i][0]==kitchen_id]
                    data_less_items[kitchen_id] = dict({'total':len(kit_items), 'items':kit_items})
                    forecast, model, metrics, residuals = [None for i in range(4)]
                    print('forecast:', forecast)
                    continue
                else:
                    s_periods_train, s_type_train = self.__seasonal_periods__(train, mode=mode)
                    s_periods, s_type = self.__seasonal_periods__(data, mode=mode)
                    try:
                        m, test_pred = self.__hw__(train, s_periods=s_periods_train, s_type=s_type_train, n_periods=len(test), mode=mode)
                        model, forecast = self.__hw__(data, s_periods=s_periods, s_type=s_type, n_periods=n_periods, mode=mode)
                    except Exception as e:
                        logger('NUCLEUS_MANCIO', 'ERR', get_traceback(e))
                        logger('NUCLEUS_MANCIO', 'ERR', 'Error in update_model() for {}_{} and item_data_id={} in kitchen={} with mode={}.'.format(self.model_name, self.model_version, item_data_id, kitchen_id, mode))
                        continue

                    test_preds = pd.DataFrame({'pred':test_pred, 'actual':test.quantity}, index=test.index)
                    dates = pd.date_range(start=test.index.max()+datetime.timedelta(days=1), periods=len(forecast), freq=mode)
                    forecast = pd.DataFrame(forecast, index=dates, columns=['forecast'])
                    forecast['yhat_lower'], forecast['yhat_upper'] = self.__conf_int__(preds=forecast, std_err_data=test_preds)
                    for col in forecast.columns:
                        forecast[col] = np.where(forecast[col]<0, 0, forecast[col])
                        forecast[col] = np.int64(np.ceil(forecast[col]))
                    print(forecast)
                    forecast.index = forecast.index.strftime('%Y-%m-%d')

                    residuals = test.quantity - test_pred

                    metrics = {}
                    metrics['aic'] = m.aic
                    metrics['bic'] = m.bic
                    metrics['mse'] = mean_squared_error(test.quantity, test_pred)
                    metrics['mae'] = mean_absolute_error(test.quantity, test_pred)
                    metrics['mase'] = mase(test.quantity, test_pred)

                _model = {}
                _model['kitchenID'] = kitchen_id
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
                db_ai.mancioModels.update_one({'kitchenID': kitchen_id, 'mode': mode, 'itemDataID': item_data_id, 'modelName': self.model_name, 'modelVersion': self.model_version}, {'$set': ml_model}, upsert=True)
        print('\nitems with no past data: \n', data_less_items)
