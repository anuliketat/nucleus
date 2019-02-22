import datetime
import requests
import importlib
from flask import current_app as app

# import inspect

from app.exceptions import NoClass, NoModule

class REControl(object):
	def __init__(self, db_main, db_ai, model_name, model_version):
		self.db_main = db_main
		self.db_ai = db_ai
		self.model_class = None

		try:
			# module_name = 'ml' + '.' + model_name + '.' + model_name+'_'+model_version.replace('.', '_')
			# _ml_module = importlib.import_module(module_name)

			# Load the module - https://www.blog.pythonlibrary.org/2016/05/27/python-201-an-intro-to-importlib/
			module_name = 'app.recommender.ml' + '.' + model_name + '.' + model_name+'_'+model_version.replace('.', '_')
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
			except AttributeError:
				print('Class does not exist')
				raise NoClass(class_name)
		except Exception as e:
			print(e)
			print('Module does not exist')
			raise NoModule(module_name)

	def get_food_recommendations(self, user_id, N=-1, online=False):
		food_items = self.model_class.get_food_recommendations(user_id, N, self.db_main, self.db_ai, online)
		return {'data': {'foodItems': food_items}}

	def update_model(self):
		# Use celery or gevent
		self.model_class.update_model(self.db_main, self.db_ai)

		return {'message': 'Model has been updated successfully!'}	
