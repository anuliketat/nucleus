import datetime
import string

import numpy as np
from fuzzywuzzy import fuzz

from api.exceptions import NoModel
from utils.misc import get_traceback, logger, sort_tuple

from ..basic_model import basic_model


class fuzzy_v1_0_0(basic_model):
	def __init__(self):
		self.model_name = 'fuzzy'
		self.model_version = 'v1.0.0'

	def __get_data__(self, db_ai):
		food_names_list = []
		food_ids_list = []
		for food_item in db_ai.foodDescription.find():
			food_names_list.append(food_item.get('recepie'))
			food_ids_list.append(food_item.get('foodID'))

		return food_names_list, food_ids_list

	def __veg_nonveg__(self, query, db_ai):
		food_names_list = []
		food_ids_list = []
		if fuzz.partial_ratio(query.lower(), 'veg') >= 67 or fuzz.partial_ratio(query.lower(), 'vegeterian') >= 75:
			if fuzz.partial_ratio(query.lower(), 'non') >= 67:
				for food_item in db_ai.foodDescription.find({'type': 'Nonveg'}):
					food_names_list.append(food_item.get('recepie'))
					food_ids_list.append(food_item.get('foodID'))
				for food_item in db_ai.foodDescription.find({'type': 'eggetarian'}):
					food_names_list.append(food_item.get('recepie'))
					food_ids_list.append(food_item.get('foodID'))
			else:
				for food_item in db_ai.foodDescription.find({'type': 'veg'}):
					food_names_list.append(food_item.get('recepie'))
					food_ids_list.append(food_item.get('foodID'))
		else:
			for food_item in db_ai.foodDescription.find():
				food_names_list.append(food_item.get('recepie'))
				food_ids_list.append(food_item.get('foodID'))

		return food_names_list, food_ids_list


	def get_search_results(self, query, N, db_ai):
		food_names_list, food_ids_list = self.__veg_nonveg__(query, db_ai)
		_scores = []
		for i in range(len(food_names_list)):
			similarity = fuzz.token_set_ratio(query, food_names_list[i])
			_scores.append((food_ids_list[i], similarity))
		_scores = sort_tuple(data=_scores, sort_key=1, descending=True)

		_final_list = []
		for _s in _scores:
			_final_list.append({'foodID': _s[0], 'score': _s[1]})

		se = {}
		se['query']: query
		se['searchResults'] = _final_list
		se['modelName'] = self.model_name
		se['modelVersion'] = self.model_version
		se['createdAt'] = datetime.datetime.utcnow()

		return se
