import time
from flask import Blueprint, g, jsonify, request

from .control import REControl
from api.exceptions import NoClass, NoModule

blue_print = Blueprint('recommender', __name__, url_prefix='/recommender')

@blue_print.route('/get-food-recommendations/<user_id>', methods=['GET'])
def get_food_recommendations(user_id):
	start_time = time.time()
	N = int(request.args.get('N', -1))
	model_name = request.args.get('model', 'lsa').lower()
	model_version = request.args.get('version', 'v1.0.0')
	online = request.args.get('online', False)

	try:
		recommender_engine = REControl(g.db_main, g.db_ai, model_name, model_version)
		response = recommender_engine.get_food_recommendations(str(user_id), int(N), online)
	except NoClass as e:
		return jsonify({'message': e.__str__()}), e.http_status()
	except NoModule as e:
		return jsonify({'message': e.__str__()}), e.http_status()
	except Exception as e:
		return e
	end_time = time.time()
	print('Execution time: ' + str(end_time-start_time))
	return jsonify(response), 200

@blue_print.route('/update-model/<model_name>/<model_version>', methods=['GET'])
def update_model(model_name, model_version):
	start_time = time.time()
	try:
		recommender_engine = REControl(g.db_main, g.db_ai, model_name, model_version)
		response = recommender_engine.update_model()
	except NoClass as e:
		return jsonify({'message': e.__str__()}), e.http_status()
	except NoModule as e:
		return jsonify({'message': e.__str__()}), e.http_status()
	except Exception as e:
		return e
	end_time = time.time()
	print('Execution time: ' + str(end_time-start_time))
	return jsonify(response), 200
