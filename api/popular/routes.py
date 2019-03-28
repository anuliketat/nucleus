import time

from flask import Blueprint, g, jsonify, request

from utils.misc import get_traceback, logger

from .control import POControl

blue_print = Blueprint('popular', __name__, url_prefix='/popular')

@blue_print.route('/get-popular-items/<kitchen_id>', methods=['GET'])
def get_popular_items(kitchen_id):
	start_time = time.time()

	N = int(request.args.get('N', -1))

	try:
		popular_engine = POControl(g.db_main)
		response = popular_engine.get_popular_items(str(kitchen_id), int(N))
	except Exception as e:
		logger('NUCLEUS_POPULAR', 'ERR', get_traceback(e))
		return jsonify({'message': 'Unknown error! Please try after sometime.'}), 500

	end_time = time.time()
	logger('NUCLEUS_POPULAR', 'EXE_TIME', 'Execution time of get_popular_items() for kitchen_id={} with N={} is {}.'.format(kitchen_id, N, end_time-start_time))
	return jsonify(response), 200
