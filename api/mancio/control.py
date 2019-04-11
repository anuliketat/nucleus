import datetime
import pickle
import importlib

import requests
from flask import current_app as app

from api.exceptions import NoClass, NoForecast
from utils.misc import logger
from utils.io import model_deserialize


class MAControl(object):
	def __init__(self, db_main, db_ai, fs_ai):
		self.db_main = db_main
		self.db_ai = db_ai
		self.fs_ai = fs_ai
		self.model_class = None

	def __load_model_class__(self, model_name, model_version):
		try:
			# Load the module - https://www.blog.pythonlibrary.org/2016/05/27/python-201-an-intro-to-importlib/
			module_name = 'api.mancio.algorithms' + '.' + model_name + '.' + model_name+'_'+model_version.replace('.', '_')
			module_spec = importlib.util.find_spec(module_name)
			_ml_module = importlib.util.module_from_spec(module_spec)
			module_spec.loader.exec_module(_ml_module)

			try:
				class_name = model_name + '_' + model_version.replace('.', '_')
				_ml_class = getattr(_ml_module, class_name)
				self.model_class = _ml_class()
			except AttributeError as e:
				raise NoClass(e, class_name)
		except Exception as e:
			raise

	def get_forecast(self, model_name, model_version, item_data_id, mode):
		logger('NUCLEUS_MANCIO', 'REQ', 'get_forecast() of {}_{} called for item_data_id={} with mode={}.'.format(model_name, model_version, item_data_id, mode))

		if model_name is not None:
			if model_version is not None:
				model_query = {
					'modelName': model_name,
					'modelVersion': model_version,
					'itemDataID': item_data_id,
					'mode': mode
				}
				ml_model = self.db_ai.mancioModels.find_one(model_query, sort=[('createdAt', -1)])
			else:
				# TODO: Get the best forecast for the model for all the versions available by comparing the metrics for the given item and kitchen
				pass
		else:
			# TODO: Get the best model forecast by comparing the metrics for the given item and kitchen
			pass

		if ml_model is None:
			raise NoForecast(model_name, model_version, item_data_id)

		model_id = ml_model.get('modelID')
		model_created_at = ml_model.get('createdAt')
		ml_model = model_deserialize(self.fs_ai.get(model_id).read())
		forecast = ml_model.get('forecast').to_json()

		logger('NUCLEUS_MANCIO', 'EXE', 'Fetching forecast from model {}_{} for item_data_id={} with mode={} successful!'.format(model_name, model_version, item_data_id, mode))
		return {'data': {'demandForecast': forecast, 'modelCreatedAt': model_created_at}}

	def update_model(self, model_name, model_version, mode):
		# Use celery or gevent
		logger('NUCLEUS_MANCIO', 'REQ', 'update_model() called for: {}_{} with mode={}.'.format(model_name, model_version, mode))
		self.__load_model_class__(model_name, model_version)
		self.model_class.update_model(self.db_main, self.db_ai, self.fs_ai, mode)
		logger('NUCLEUS_MANCIO', 'EXE', 'Update of the model: {}_{} with mode={} successful!'.format(model_name, model_version, mode))
		return {'message': 'Model has been updated successfully!'}
