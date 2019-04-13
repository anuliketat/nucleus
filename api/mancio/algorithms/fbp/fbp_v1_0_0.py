import datetime
import pickle

import numpy as np
import pandas as pd
from bson.binary import Binary
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics

from api.exceptions import NoModel
from utils.mase import mase
from utils.misc import get_traceback, logger
from utils.io import model_serialize

from ..basic_model.basic_model import basic_model


class fbp_v1_0_0(basic_model):
    def __init__(self):
        self.model_name = 'fbp'
        self.model_version = 'v1.0.0'

    def __get_data__(self, item_data_id, db_main, kitchen_id=2, mode='D'):
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
        all_orders = all_orders[all_orders.item_data_id!=None]
        all_orders.time = pd.to_datetime(all_orders.time, dayfirst=True, format='%Y-%m-%d %H:%M').dt.date
        all_orders.time = pd.to_datetime(all_orders.time)

        for k in all_orders.kitchen_id.unique():
            kit_id[k] = all_orders.loc[all_orders.kitchen_id==k]
        orders = kit_id[kitchen_id]
        item = orders.loc[orders['item_data_id']==item_data_id].groupby('time')[['quantity']].sum()
        data = item.resample(mode).sum().fillna(0)
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

    def __fbp__(self, ts_data, n_periods=2):
        """
            ts_data - time series data with time column as 'ds' and value as 'y'
        """
        holidays = self.__df_holidays__()
        model = Prophet(holidays=holidays, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False) #conf.int = 80%
        model.fit(ts_data)
        future = model.make_future_dataframe(periods=n_periods, include_history=False)
        forecast = model.predict(future)
        forecast = forecast[['ds', 'yhat_lower', 'yhat_upper', 'yhat']]
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                forecast[col] = np.where(forecast[col]<0, 0, forecast[col])
                forecast[col] = np.int64(np.ceil((forecast[col])))

        return model, forecast

    def update_model(self, db_main, db_ai, fs_ai, mode='D'):
        item_data_ids, kitchen_ids = [], []
        for d in db_main.ordered_items.find():
            item_data_ids.append(d.get('item_data_id'))
            kitchen_ids.append(d.get('kitchen_id'))
        item_data_ids = [i for i in item_data_ids if i]

        for kitchen_id in set(kitchen_ids):
            for item_data_id in set(item_data_ids):
                print('\nItem ID:', item_data_id)
                train, test, data = self.__get_data__(item_data_id, db_main, kitchen_id=kitchen_id, mode=mode)
                try:
                    m, test_pred = self.__fbp__(train, n_periods=len(test))
                    model, forecast = self.__fbp__(data)
                except Exception as e:
                    logger('NUCLEUS_MANCIO', 'ERR', get_traceback(e))
                    logger('NUCLEUS_MANCIO', 'ERR', 'Error in update_model() for {}_{} and item_id={} with mode={}.'.format(self.model_name, self.model_version, item_data_id, mode))
                    continue

                residuals = test.y - test_pred.yhat
                metrics = {}
                #metrics['aic'] = m.aic
                #metrics['bic'] = m.bic
                metrics['mse'] = mean_squared_error(test.y, test_pred.yhat)
                metrics['mae'] = mean_absolute_error(test.y, test_pred.yhat)
                metrics['mase'] = mase(test.y, test_pred.yhat)

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
