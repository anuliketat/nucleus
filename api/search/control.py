import datetime
import importlib

import requests
from flask import current_app as app

from api.exceptions import NoClass
from utils.misc import logger


class SEControl(object):
	def __init__(self, db_ai, model_name, model_version):
		self.db_ai = db_ai
		self.model_class = None

		try:
			module_name = 'api.search.algorithms' + '.' + model_name + '.' + model_name+'_'+model_version.replace('.', '_')
			module_spec = importlib.util.find_spec(module_name)
			_module = importlib.util.module_from_spec(module_spec)
			module_spec.loader.exec_module(_module)

			try:
				class_name = model_name + '_' + model_version.replace('.', '_')
				_class = getattr(_module, class_name)
				self.model_class = _class()
			except AttributeError as e:
				raise NoClass(e, class_name)
		except Exception as e:
			raise

	def get_search_results(self, query, N=-1):
		logger('NUCLEUS_SEARCH', 'REQ', 'get_search_results() of {}_{} called for query={} with N={}.'.format(self.model_class.model_name, self.model_class.model_version, query, N))
		try:
			se = self.model_class.get_search_results(query, N, self.db_ai)
		except Exception:
			raise

		search_results = se['searchResults']
		num_results = len(search_results)
		model_created_at = se['modelCreatedAt']

		logger('NUCLEUS_SEARCH', 'EXE', 'Fetching search results from model {}_{} for query={} with N={} successful!'.format(self.model_class.model_name, self.model_class.model_version, query, N))
		return {'data': {'searchResults': search_results, 'modelCreatedAt': model_created_at}}

	def update_model(self):
		logger('NUCLEUS_SEARCH', 'REQ', 'update_model() called for: {}_{}.'.format(self.model_class.model_name, self.model_class.model_version))
		self.model_class.update_model(self.db_ai)
		logger('NUCLEUS_SEARCH', 'EXE', 'Update of the model: {}_{} successful!'.format(self.model_class.model_name, self.model_class.model_version))
		return {'message': 'Model has been updated successfully!'}


