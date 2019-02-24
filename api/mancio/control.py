import datetime
import requests
import importlib
from flask import current_app as app

# import inspect

from api.exceptions import NoClass

class MAControl(object):
	def __init__(self, db_main, db_ai, fs_ai, model_name, model_version):
		self.db_main = db_main
		self.db_ai = db_ai
		self.fs_ai = fs_ai
		self.model_class = None

		try:
			# module_name = 'ml' + '.' + model_name + '.' + model_name+'_'+model_version.replace('.', '_')
			# _ml_module = importlib.import_module(module_name)

			# Load the module - https://www.blog.pythonlibrary.org/2016/05/27/python-201-an-intro-to-importlib/
			module_name = 'api.mancio.ml' + '.' + model_name + '.' + model_name+'_'+model_version.replace('.', '_')
			module_spec = importlib.util.find_spec(module_name)
			_ml_module = importlib.util.module_from_spec(module_spec)
			module_spec.loader.exec_module(_ml_module)

			# for name, obj in inspect.getmembers(_ml_module):
			# 	print(name)
			# 	if inspect.isclass(obj):
			# 		print(obj)

			try:
				class_name = model_name + '_' + model_version.replace('.', '_')
				_ml_class = getattr(_ml_module, class_name)
				self.model_class = _ml_class()
			except AttributeError as e:
				raise NoClass(e, class_name)
		except Exception as e:
			raise

	def update_model(self):
		# Use celery or gevent
		self.model_class.update_model(self.db_main, self.db_ai, self.fs_ai)
		return {'message': 'Model has been updated successfully!'}
