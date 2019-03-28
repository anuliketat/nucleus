import datetime
import importlib

import requests
from flask import current_app as app

from api.exceptions import NoClass
from utils.misc import logger


class MAControl(object):
	def __init__(self, db_main, db_ai, fs_ai, model_name, model_version):
		self.db_main = db_main
		self.db_ai = db_ai
		self.fs_ai = fs_ai
		self.model_class = None

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

	def update_model(self, mode):
		# Use celery or gevent
		logger('NUCLEUS_MANCIO', 'REQ', 'update_model() called for: {}_{} with mode={}.'.format(self.model_class.model_name, self.model_class.model_version, mode))
		self.model_class.update_model(self.db_main, self.db_ai, self.fs_ai, mode)
		logger('NUCLEUS_MANCIO', 'EXE', 'Update of the model: {}_{} with mode={} successful!'.format(self.model_class.model_name, self.model_class.model_version, mode))
		return {'message': 'Model has been updated successfully!'}
