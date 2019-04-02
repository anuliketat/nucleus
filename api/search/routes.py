import time

from flask import Blueprint, g, jsonify, request

from api.exceptions import NoClass
from utils.misc import get_traceback, logger

from .control import SEControl

blue_print = Blueprint('search', __name__, url_prefix='/search')

@blue_print.route('/get-search-results/<query>', methods=['GET'])
def get_search_results(query):
	start_time = time.time()

	N = int(request.args.get('N', -1))
	model_name = request.args.get('model', 'fuzzy').lower()
	model_version = request.args.get('version', 'v2.0.0')

	try:
		search_engine = SEControl(g.db_ai, model_name, model_version)
		response = search_engine.get_search_results(str(query), int(N))
	except NoClass as e:
		logger('NUCLEUS_SEARCH', 'ERR', e.traceback())
		logger('NUCLEUS_SEARCH', 'ERR', e.__str__())
		return jsonify({'message': e.__str__()}), e.http_status()
	except Exception as e:
		logger('NUCLEUS_SEARCH', 'ERR', get_traceback(e))
		return jsonify({'message': 'Unknown error! Please try after sometime.'}), 500

	end_time = time.time()
	logger('NUCLEUS_SEARCH', 'EXE_TIME', 'Execution time of get_search_results() for query={} with N={} is {}.'.format(query, N, end_time-start_time))
	return jsonify(response), 200

@blue_print.route('/update-model/<model_name>/<model_version>', methods=['GET'])
def update_model(model_name, model_version):
	start_time = time.time()

	try:
		search_engine = SEControl(g.db_ai, model_name, model_version)
		response = search_engine.update_model()
	except NoClass as e:
		logger('NUCLEUS_SEARCH', 'ERR', e.traceback())
		logger('NUCLEUS_SEARCH', 'ERR', e.__str__())
		return jsonify({'message': e.__str__()}), e.http_status()
	except Exception as e:
		logger('NUCLEUS_SEARCH', 'ERR', get_traceback(e))
		return jsonify({'message': 'Unknown error! Please try after sometime.'}), 500

	end_time = time.time()
	logger('NUCLEUS_SEARCH', 'EXE_TIME', 'Execution time of update_model() for {}_{} is {}.'.format(model_name, model_version, end_time-start_time))
	return jsonify(response), 200