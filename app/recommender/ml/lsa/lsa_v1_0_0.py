import datetime
from bson.binary import Binary

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse.linalg import svds

import string 
import pickle

from app.exceptions import NoStartupRecommendations
from utils.metrics import cosine_sim
from utils.misc import sort_tuple

from ..basic_model import basic_model

class lsa_v1_0_0(basic_model):
	def __init__(self):
		self.model_name = 'lsa'
		self.model_version = 'v1.0.0'

	# Need implementation
	def __get_max_k__(self):
		return 100

	# Need implementation
	def __get_data__(self, db_ai):
		food_list = []
		food_ids_list = []

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
			details = " ".join(details)

			stemmed_data.append(details)

		transformer = TfidfVectorizer()
		tf_idf = transformer.fit_transform(stemmed_data).T

		U, S, Vt = svds(tf_idf, k=self.__get_max_k__())

		S = np.diag(S)
		food_profiles = S.dot(Vt)
		return food_profiles

	def __model_serialize__(self, model):
		return Binary(pickle.dumps(model, protocol=2))
		# return pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)

	def __model_deserialize__(self, model):
		return pickle.loads(model)

	# Need implementation
	def __update_recommendations__(self, food_profiles, db_main, db_ai):
		print('__update_recommendations__() function called')
		return 0

	# Need implementation
	def get_food_recommendations(self, user_id, N, db_main, db_ai, online=False):
		print('get_food_recommendations() function called')
		return 0

	# Need implementation
	def update_model(self, db_main, db_ai):
		print('update_model() function called')
