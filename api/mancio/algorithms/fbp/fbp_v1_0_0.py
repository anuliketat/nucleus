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

from ..basic_model.basic_model import basic_model


class fbp_v1_0_0(basic_model):
    def __init__(self):
        self.model_name = 'fbp'
        self.model_version = 'v1.0.0'

    def __model_serialize__(self, model):
        return Binary(pickle.dumps(model, protocol=2))

    def __model_deserialize__(self, model):
        return pickle.loads(model)

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

    def get_forecast(self, db_main, db_ai, fs_ai, item_data_id, mode='D'):
        ml_model = db_ai.models.find_one(
                                        {'modelName': self.model_name,
                                        'modelVersion': self.model_version,
                                        'itemDataID': item_data_id,
                                        'mode': mode},
                                        sort=[('createdAt', -1)]
                                        )
        if ml_model is None:
            raise NoModel(self.model_name, self.model_version)

        model_id = ml_model.get('modelID')
        _model_created_at = ml_model.get('createdAt')
        ml_model_data = self.__model_deserialize__(fs_ai.get(model_id).read())
        items_data = ml_model_data.get('data')
        forecast = ml_model_data.get('forecast')

        manc = {}
        manc['itemDataID'] = item_data_id
        manc['kitchenID'] = ml_model.get('kitchen')
        manc['mode'] = ml_model.get('mode')
        manc['demandForecast'] = forecast.to_json()
        manc['modelName'] = self.model_name
        manc['modelVersion'] = self.model_version
        manc['createdAt'] = datetime.datetime.utcnow()
        manc['modelCreatedAt'] = _model_created_at

        return manc

    def update_model(self, db_main, db_ai, fs_ai, mode='D'):
        item_data_ids, kitchen_ids = [], []
        for d in db_main.ordered_items.find():
            item_data_ids.append(d.get('item_data_id'))
            kitchen_ids.append(d.get('kitchen_id'))
        item_data_ids = [i for i in item_data_ids if i]

        for k in set(kitchen_ids):
            for item_id in set(item_data_ids):
                print('\nItem ID:', item_id)
                train, test, data = self.__get_data__(item_id, db_main, kitchen_id=k, mode=mode)
                try:
                    m, test_pred = self.__fbp__(train, n_periods=len(test))
                    model, forecast = self.__fbp__(data)
                except Exception as e:
                    logger('NUCLEUS_MANCIO', 'ERR', get_traceback(e))
                    logger('NUCLEUS_MANCIO', 'ERR', 'Error in update_model() for {}_{} and item_id={} with mode={}.'.format(self.model_name, self.model_version, item_id, mode))
                    continue

                residuals = test.y - test_pred.yhat
                metrics = {}
                #metrics['aic'] = m.aic
                #metrics['bic'] = m.bic
                metrics['mse'] = mean_squared_error(test.y, test_pred.yhat)
                metrics['mae'] = mean_absolute_error(test.y, test_pred.yhat)
                metrics['mase'] = mase(test.y, test_pred.yhat)

                _model = {}
                _model['kitchen'] = k
                _model['itemDataID'] = item_id
                _model['data'] = data
                _model['forecast'] = forecast
                _model['residuals'] = residuals
                _model['model'] = model
                model_id = fs_ai.put(self.__model_serialize__(_model))

                ml_model = {}
                ml_model['modelName'] = self.model_name
                ml_model['modelVersion'] = self.model_version
                ml_model['modelID'] = model_id
                ml_model['itemDataID'] = item_id
                ml_model['metrics'] = metrics
                ml_model['mode'] = mode
                ml_model['createdAt'] = datetime.datetime.utcnow()
                db_ai.models.insert_one(ml_model)
                print(_model['forecast'])
