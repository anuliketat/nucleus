import datetime
import pickle

import numpy as np
import pandas as pd
#import scipy
from bson.binary import Binary
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa import arima_model
from statsmodels.tsa.statespace import sarimax

from utils.mase import mase
from utils.misc import get_traceback, logger
from utils.z_score import z_score
from utils.io import model_serialize

from ..basic_model.basic_model import basic_model


class aa_v1_0_0(basic_model):
    def __init__(self):
        self.model_name = 'aa'
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
        quantity, date = [], []
        for data in db_main.ordered_items.aggregate(pipeline):
            date.append(data.get('_id'))
            quantity.append(data.get('quant'))

        item = pd.DataFrame({'quantity':quantity, 'timestamp':date})
        item.dropna(axis=0, inplace=True)
        item.timestamp = pd.to_datetime(item.timestamp, dayfirst=True, format='%Y-%m-%d %H:%M').dt.date #sparse date from datetime
        item.timestamp = pd.to_datetime(item.timestamp)
        item.set_index('timestamp', inplace=True)
        print('item records: ', item.shape[0])

        data = item.resample(mode).sum().fillna(0)
        data['dayOfWeek'] = pd.DatetimeIndex(data.index).weekday+1
        data['month'] = pd.DatetimeIndex(data.index).month
        train = data[:int(0.9*(len(data)))]
        test = data[int(0.9*(len(data))):]
        for col in item.columns[1:]:
            item[col] = item[col].astype('object')

        return train, test, data

    def __future_df__(self, ts_data, n_periods, mode='D'):
        dates = pd.date_range(start=ts_data.index.max()+datetime.timedelta(days=1), periods=n_periods, freq=mode)
        future_df = pd.DataFrame({'dayOfWeek':0, 'month':0}, index=dates)
        future_df['dayOfWeek'] = pd.DatetimeIndex(future_df.index).weekday+1
        future_df['month'] = pd.DatetimeIndex(future_df.index).month
        for col in future_df.columns:
            future_df[col] = future_df[col].astype('object')

        return future_df

    def __auto_arima__(self, ts_data, fit_data, validation_size, n_periods, mode='D', validation=False):
        """
            ts_data - time series data with datetime index and a column 'quantity'
            n_periods - #periods to forecast
            returns model and forecasted values for the period.
        """
        m = (7 if mode=='D' else 1) #no.of periodic cycles
        model = auto_arima(ts_data.quantity, m=m, error_action='ignore', suppress_warnings=True, start_p=0, start_q=0, max_p=4, max_q=4,
                            start_P=0, start_Q=0, seasonal=False, stepwise=True, n_jobs=1,
                            random=True, out_of_sample_size=validation_size, scoring='mae')
        exog_vars = np.column_stack((ts_data.dayOfWeek, ts_data.month)) #other exogenous variables to train
        model.fit(fit_data.quantity, exogenous=exog_vars[:len(fit_data)])
        if validation:
            forecast = model.predict(exogenous=exog_vars[len(fit_data):], n_periods=n_periods)
        else:
            future_df = self.__future_df__(ts_data, n_periods=n_periods, mode=mode)
            exog_vars_future = np.column_stack((future_df.dayOfWeek, future_df.month))
            forecast = model.predict(exogenous=exog_vars_future, n_periods=n_periods)

        return model, forecast

    def __conf_int__(self, preds_df, std_err_df, alpha=0.20):
        """
            preds_df - forecasted time series dataframe
            st_err_df - test data and predicted test dataframe
            alpha - 0 to 1. Confidence intervals for the forecasted values. Default is 80%
            returns lower and upper conf intervals for the forecasts
        """
        z_scr = z_score(alpha)
        if std_err_df.isnull().values.any():
            lower, upper = np.NaN, np.NaN
        else:
            lower, upper = [], []
            for i in preds_df.forecast:
                a = i-z_scr*((mean_squared_error(std_err_df.actual, std_err_df.pred))**0.5)
                b = i+z_scr*((mean_squared_error(std_err_df.actual, std_err_df.pred))**0.5)
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

        data_less_items = {}
        no_data_items = [] #a blacklist for items with no data
        for kitchen_id in sorted(set(kitchen_ids)):
            items = sorted(orders[orders.kit_id==kitchen_id].item_data_id.unique())
            items = [int(i) for i in items if i]
            print('-'*45, 'Kitchen ID: {}'.format(kitchen_id), '-'*45)
            for item_data_id in items:
                print('\nKitchen: {} | Item Data ID: {} | Mode: {} | Model: {}'.format(kitchen_id, item_data_id, mode, self.model_name.upper()))
                train, test, data = self.__get_data__(item_data_id, db_main, kitchen_id, mode=mode)
                if len(data) <= 3: # for items with no data, forecast is none
                    no_data_items.append(tuple((kitchen_id, item_data_id)))
                    kit_items = [no_data_items[i][1] for i in range(len(no_data_items)) if no_data_items[i][0]==kitchen_id]
                    data_less_items[kitchen_id] = dict({'total':len(kit_items), 'items':kit_items})
                    forecast, model, metrics, residuals = [None for i in range(4)]
                    print('forecast:', forecast)
                    continue
                else:
                    try:
                        m, test_preds = self.__auto_arima__(data, train, validation_size=len(test), n_periods=len(test), validation=True)
                        model, forecast = self.__auto_arima__(data, data, validation_size=len(test), n_periods=int(n_periods))
                    except Exception as e:
                        logger('NUCLEUS_MANCIO', 'ERR', get_traceback(e))
                        logger('NUCLEUS_MANCIO', 'ERR', 'Error in update_model() for {}_{} and item_id={} with mode={}.'.format(self.model_name, self.model_version, item_data_id, mode))
                        continue

                    test_preds_df = pd.DataFrame({'pred':test_preds, 'actual':test.quantity}, index=test.index)
                    dates = pd.date_range(start=test.index.max()+datetime.timedelta(days=1), periods=len(forecast), freq=mode)
                    forecast = pd.DataFrame(forecast, index=dates, columns=['forecast'])
                    forecast['yhat_lower'], forecast['yhat_upper'] = self.__conf_int__(preds_df=forecast, std_err_df=test_preds_df)
                    for col in forecast.columns:
                        forecast[col] = np.where(forecast[col]<0, 0, forecast[col])
                        forecast[col] = np.int32(np.ceil(forecast[col]))
                    print(forecast)
                    forecast.index = forecast.index.strftime('%Y-%m-%d')

                    residuals = m.resid()

                    metrics = {}
                    metrics['aic'] = m.aic()
                    metrics['bic'] = m.bic()
                    metrics['mse'] = np.NaN if test_preds_df.isnull().values.any() else mean_squared_error(test_preds_df.actual, test_preds_df.pred)
                    metrics['mae'] = np.NaN if test_preds_df.isnull().values.any() else mean_absolute_error(test_preds_df.actual, test_preds_df.pred)
                    metrics['mase'] = np.NaN if test_preds_df.isnull().values.any() else mase(test_preds_df.actual, test_preds_df.pred)

                _model = {}
                _model['kitchen'] = kitchen_id
                _model['itemDataID'] = item_data_id
                _model['data'] = data
                _model['forecast'] = forecast
                _model['residuals'] = residuals
                _model['model'] = model
                #_model['summary'] = m.summary()
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