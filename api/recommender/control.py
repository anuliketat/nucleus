import datetime
import importlib
# import inspect

import requests
from flask import current_app as app

from api.exceptions import NoClass
from utils.misc import logger


class REControl(object):
	def __init__(self, db_main, db_ai, fs_ai, model_name, model_version):
		self.db_main = db_main
		self.db_ai = db_ai
		self.fs_ai = fs_ai
		self.model_class = None

		try:
			# module_name = 'ml' + '.' + model_name + '.' + model_name+'_'+model_version.replace('.', '_')
			# _ml_module = importlib.import_module(module_name)

			# Load the module - https://www.blog.pythonlibrary.org/2016/05/27/python-201-an-intro-to-importlib/
			module_name = 'api.recommender.ml' + '.' + model_name + '.' + model_name+'_'+model_version.replace('.', '_')
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

	def get_popular_items(self, kitchen_id, N=-1):
		logger('NUCLEUS_RECOMMENDER', 'REQ', 'get_popular_items() called for kitchen_id={} with N={}.'.format(kitchen_id, N))
		pipeline = [
            {
                "$match": {
                    "item_data_id": {"$ne": None},
                    "kitchen_id": {"$eq": kitchen_id},
                }
            },
            {
                "$group": {
                    "_id": {
                        "item_data_id": "$item_data_id",
                        "kitchen_id": "$kitchen_id",
                    },
                    "count": {"$sum": 1},
                }
            },
            # {"$match": {"count": {"$gt": 0}}},
            {"$sort": {"count": -1}},
        ]
		popular_items = list(self.db_main.ordered_items.aggregate(pipeline))
		num_items = len(popular_items)
		if N != -1 and N <= num_items:
			popular_items = popular_items[0:N]

		logger('NUCLEUS_RECOMMENDER', 'EXE', 'Fetching popular items for kitchen_id={} with N={} successful!'.format(kitchen_id, N))
		return {'data': {'popularItems': popular_items}}

	def get_food_recommendations(self, user_id, N=-1):
		logger('NUCLEUS_RECOMMENDER', 'REQ', 'get_food_recommendations() of {}_{} called for user_id={} with N={}.'.format(self.model_class.model_name, self.model_class.model_version, user_id, N))
		try:
			reco = self.model_class.get_food_recommendations(user_id, N, self.db_main, self.db_ai, self.fs_ai)
		except Exception:
			raise

		food_recommendations = reco['foodRecommendations']
		num_recommendations = len(food_recommendations)
		model_created_at = reco['modelCreatedAt']
		if N != -1 and N <= num_recommendations:
			food_recommendations = food_recommendations[0:N]

		logger('NUCLEUS_RECOMMENDER', 'EXE', 'Fetching food recommendations from model {}_{} for user_id={} with N={} successful!'.format(self.model_class.model_name, self.model_class.model_version, user_id, N))
		return {'data': {'foodRecommendations': food_recommendations, 'modelCreatedAt': model_created_at}}

	def update_model(self):
		# Use celery or gevent
		logger('NUCLEUS_RECOMMENDER', 'REQ', 'update_model() called for: {}_{}.'.format(self.model_class.model_name, self.model_class.model_version))
		self.model_class.update_model(self.db_main, self.db_ai, self.fs_ai)
		logger('NUCLEUS_RECOMMENDER', 'EXE', 'Update of the model: {}_{} successful!'.format(self.model_class.model_name, self.model_class.model_version))
		return {'message': 'Model has been updated successfully!'}
