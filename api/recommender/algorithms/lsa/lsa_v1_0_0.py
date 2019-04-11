import datetime
import pickle
import string

import numpy as np
from bson.binary import Binary
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer

from api.exceptions import NoModel
from utils.io import model_deserialize, model_serialize
from utils.metrics import cosine_sim
from utils.misc import get_traceback, logger, sort_tuple

from ..basic_model import basic_model


class lsa_v1_0_0(basic_model):
	def __init__(self):
		self.model_name = 'lsa'
		self.model_version = 'v1.0.0'

	def __get_max_k__(self):
		return 95

	def __max_past_orders__(self):
		return 15

	def __get_data__(self, db_ai):
		food_list = []
		food_ids_list = []
		for food_item in db_ai.foodDescription.find():
			food_ids_list.append(food_item.get('foodID'))

			data = ''
			data += str(food_item.get('recipie')) + ' '
			data += str(food_item.get('recipieDescription')) + ' '
			data += str(food_item.get('ingredients')) + ' '
			data += str(food_item.get('primaryIngredients')) + ' '
			data += str(food_item.get('secondarayIngredients')) + ' '
			data += str(food_item.get('type')) + ' '
			data += str(food_item.get('cuisine')) + ' '
			data += str(food_item.get('category')) + ' '
			data += str(food_item.get('meal')) + ' '
			data += str(food_item.get('spicy')) + 'spicy' + ' '
			data += str(food_item.get('taste')) + ' '
			data += str(food_item.get('primaryColors')) + ' '
			data += str(food_item.get('secondaryColors')) + ' '
			data += str(food_item.get('cookingMethod')) + ' '
			data += str(food_item.get('servedAs')) + ' '
			data += str(food_item.get('calories')) + 'calories'

			food_list.append(data)

		return (food_ids_list, food_list)

	def __pre_processor__(self, details):
		ignore_chars = ''',:"&])-([''!/+.'''
		stemmer = PorterStemmer()
		# lemmetizer = WordNetLemmatizer()

		details = details.translate(string.punctuation).lower()
		details = word_tokenize(details)
		details = [stemmer.stem(word) for word in details if not (word in stopwords.words('english') or word in ignore_chars or word == 'none' or word == 'nil' or word == 'null')]
		# details = [lemmetizer.lemmatize(word) for word in details if not (word in stopwords.words('english') or word in ignore_chars or word == 'none' or word == 'nil' or word == 'null')]
		details = ' '.join(details)

		details = details.replace("'", "")
		details = details.replace('.', '')
		details = details.replace('/', '')

		return details

	def __lsa__(self, food_list):
		stemmed_data = []
		for i in range(0, len(food_list)):
			details = food_list[i]
			details = self.__pre_processor__(details)
			stemmed_data.append(details)

		transformer = TfidfVectorizer()
		tf_idf = transformer.fit_transform(stemmed_data).T

		_, S, Vt = svds(tf_idf, k=self.__get_max_k__())

		S = np.diag(S)
		food_profiles = S.dot(Vt)
		return food_profiles

	def get_food_recommendations(self, user_id, N, db_main, db_ai, fs_ai):
		# TODO: Add a check whether the user actually exists or not
		ordered_item_data_ids = []

		# TODO: Use MongoDB aggregation pipeline
		# TODO: Add a check if '__max_past_orders__' requirement is satisfied or not
		for order in db_main.orders.find({'user_id': user_id}, sort=[('timestamp', -1)]).limit(self.__max_past_orders__()):
			for item in order.get('ordered_items', []):
				ordered_item_data_ids.append(item.get('item_data_id'))

		ml_model = db_ai.recommenderModels.find_one({'modelName': self.model_name, 'modelVersion': self.model_version}, sort=[('createdAt', -1)])
		if ml_model is None:
			raise NoModel(self.model_name, self.model_version)

		model_id = ml_model.get('modelID')
		_model_created_at = ml_model.get('createdAt')
		ml_model = model_deserialize(fs_ai.get(model_id).read())
		food_profiles = ml_model.get('foodProfiles')
		food_ids_list = ml_model.get('foodIDsList')
		taste_profile = np.zeros(food_profiles[:, 0].shape)

		count = 0
		for item_data_id in ordered_item_data_ids:
			try:
				index = food_ids_list.index(item_data_id)
				taste_profile += food_profiles[:, index]
				count += 1
			except ValueError:
				logger('NUCLEUS_RECOMMENDER', 'WARN', 'Item with item_data_id: {} does not exist!'.format(item_data_id))

		if count > 0:
			taste_profile = taste_profile/count

		_scores = []
		for i in range(0, len(food_ids_list)):
			similarity = cosine_sim(taste_profile, food_profiles[:, i])
			_scores.append((str(food_ids_list[i]), similarity))
		_scores = sort_tuple(data=_scores, sort_key=1, descending=True)

		_final_list = []
		for _s in _scores:
			_final_list.append({'itemDataID': _s[0], 'score': _s[1]})

		reco = {}
		reco['userID']: user_id
		reco['foodRecommendations'] = _final_list
		reco['modelName'] = self.model_name
		reco['modelVersion'] = self.model_version
		reco['createdAt'] = datetime.datetime.utcnow()
		reco['modelCreatedAt'] = _model_created_at

		db_ai.foodRecommendations.update_one({'userID': user_id, 'modelName': self.model_name, 'modelVersion': self.model_version}, {'$set': reco}, upsert=True)
		return reco

	def update_model(self, db_main, db_ai, fs_ai):
		food_ids_list, food_list = self.__get_data__(db_ai)
		food_profiles = self.__lsa__(food_list)

		_model = {}
		_model['foodProfiles'] = food_profiles
		_model['foodIDsList'] = food_ids_list
		model_id = fs_ai.put(model_serialize(_model))

		ml_model = {}
		ml_model['modelName'] = self.model_name
		ml_model['modelVersion'] = self.model_version
		ml_model['modelID'] = model_id
		ml_model['createdAt'] = datetime.datetime.utcnow()
		db_ai.recommenderModels.update_one({'modelName': self.model_name, 'modelVersion': self.model_version}, {'$set': ml_model}, upsert=True)
