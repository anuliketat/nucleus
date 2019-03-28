import datetime
import importlib

import requests
from flask import current_app as app

from utils.misc import logger


class POControl(object):
	def __init__(self, db_main):
		self.db_main = db_main

	def get_popular_items(self, kitchen_id, N=-1):
		logger('NUCLEUS_POPULAR', 'REQ', 'get_popular_items() called for kitchen_id={} with N={}.'.format(kitchen_id, N))
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

		logger('NUCLEUS_POPULAR', 'EXE', 'Fetching popular items for kitchen_id={} with N={} successful!'.format(kitchen_id, N))
		return {'data': {'popularItems': popular_items}}
