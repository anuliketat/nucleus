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

from utils.metrics import cosine_sim
from utils.misc import get_traceback, logger, sort_tuple

from ..basic_model import basic_model


class lsa_v1_0_0(basic_model):
	def __init__(self):
		self.model_name = 'lsa'
		self.model_version = 'v1.0.0'

	# Need implementation
	def __get_max_k__(self):
		return 95

	# Need implementation
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

	def __tf_idf__(self, food_list):
		ignore_chars = ''',:"&])-([''!/+.'''
		stemmer = PorterStemmer()

		stemmed_data = []
		for i in range(0, len(food_list)):
			details = food_list[i]
			details = details.translate(string.punctuation).lower()
			details = word_tokenize(details)
			details = [stemmer.stem(word) for word in details if not (word in stopwords.words('english') or word in ignore_chars)]
			details = ' '.join(details)

			stemmed_data.append(details)

		transformer = TfidfVectorizer()
		tf_idf = transformer.fit_transform(stemmed_data).T

		U, S, Vt = svds(tf_idf, k=self.__get_max_k__())
		# with open('./internal_testing/S.pkl', 'wb') as fh:
			# pickle.dump(S, fh)

		S = np.diag(S)
		food_profiles = S.dot(Vt)
		return food_profiles

	def __model_serialize__(self, model):
		return Binary(pickle.dumps(model, protocol=2))
		# return pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)

	def __model_deserialize__(self, model):
		return pickle.loads(model)

	# Need implementation
	def __update_recommendations__(self, food_profiles, _model_created_at, db_main, db_ai):
		print('__update_recommendations__() function called')
		return 0

	# Need implementation
	def get_food_recommendations(self, user_id, N, db_main, db_ai, online=False):
		print('get_food_recommendations() function called')
		return 0

	# Need implementation
	def update_model(self, db_main, db_ai, fs_ai):
		logger('NUCLEUS_RECOMMENDER', 'REQ', 'update_model() called for: {}_{}.'.format(self.model_name, self.model_version))

		food_ids_list, food_list = self.__get_data__(db_ai)
		food_profiles = self.__tf_idf__(food_list)

		_model = {}
		_model['foodProfiles'] = food_profiles
		_model['foodIDsList'] = food_ids_list
		model_id = fs_ai.put(self.__model_serialize__(_model))

		ml_model = {}
		ml_model['modelName'] = self.model_name
		ml_model['modelVersion'] = self.model_version
		ml_model['modelID'] = model_id
		ml_model['createdAt'] = datetime.datetime.utcnow()
		db_ai.models.insert_one(ml_model)

		print(food_profiles)
		_model_created_at = ml_model['createdAt']
		self.__update_recommendations__(food_profiles, _model_created_at, db_main, db_ai)
		logger('NUCLEUS_RECOMMENDER', 'EXE', 'Update of the model: {}_{} successful!'.format(self.model_name, self.model_version))
