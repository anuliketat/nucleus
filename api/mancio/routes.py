import time

from flask import Blueprint, g, jsonify, request

from api.exceptions import NoClass, NoForecast
from utils.misc import get_traceback, logger

from .control import MAControl

blue_print = Blueprint('mancio', __name__, url_prefix='/mancio')

@blue_print.route('/get-forecast/<item_data_id>/<kitchen_id>', defaults={'model_name': None, 'model_version': None})
@blue_print.route('/get-forecast/<model_name>/<model_version>/<item_data_id>/<kitchen_id>', methods=['GET'])
def get_forecast(model_name, model_version, item_data_id, kitchen_id):
	start_time = time.time()

	allowed_modes = ['D', 'W', 'M']
	mode = request.args.get('mode', 'D').upper()
	if mode not in allowed_modes:
		mode = 'D'
	n_periods = request.args.get('p')
	if n_periods is None:
		n_periods = 5

	try:
		mancio_engine = MAControl(g.db_main, g.db_ai, g.fs_ai)
		response = mancio_engine.get_forecast(model_name, model_version, int(item_data_id), str(kitchen_id), mode, n_periods)
	except NoClass as e:
		logger('NUCLEUS_MANCIO', 'ERR', get_traceback(e))
		logger('NUCLEUS_MANCIO', 'ERR', e.__str__())
		return jsonify({'message': e.__str__()}), e.http_status()
	except NoForecast as e:
		logger('NUCLEUS_MANCIO', 'ERR', get_traceback(e))
		logger('NUCLEUS_MANCIO', 'ERR', e.__str__())
		return jsonify({'message': e.__str__()}), e.http_status()
	except Exception as e:
		logger('NUCLEUS_MANCIO', 'ERR', get_traceback(e))
		return jsonify({'message': 'Unknown error! Please try after sometime.'}), 500

	end_time = time.time()
	logger('NUCLEUS_MANCIO', 'EXE_TIME', 'Execution time of get_forecast() for item_data_id={} in kitchen={} with mode={} for {} periods is {}.'.format(item_data_id, kitchen_id, mode, n_periods, end_time-start_time))
	return jsonify(response), 200

@blue_print.route('/update-model', defaults={'model_name': None, 'model_version': None})
@blue_print.route('/update-model/<model_name>/<model_version>', methods=['GET'])
def update_model(model_name, model_version):
	start_time = time.time()

	allowed_modes = ['D', 'W', 'M']
	mode = request.args.get('mode', 'D').upper()
	if mode not in allowed_modes:
		mode = 'D'
	n_periods = request.args.get('p')
	if n_periods is None:
		n_periods = 5

	try:
		mancio_engine = MAControl(g.db_main, g.db_ai, g.fs_ai)
		response = mancio_engine.update_model(model_name, model_version, n_periods, mode)
	except NoClass as e:
		logger('NUCLEUS_MANCIO', 'ERR', get_traceback(e))
		logger('NUCLEUS_MANCIO', 'ERR', e.__str__())
		return jsonify({'message': e.__str__()}), e.http_status()
	except Exception as e:
		logger('NUCLEUS_MANCIO', 'ERR', get_traceback(e))
		return jsonify({'message': 'Unknown error! Please try after sometime.'}), 500

	end_time = time.time()
	logger('NUCLEUS_MANCIO', 'EXE_TIME', 'Execution time of update_model() for {}_{} is {}.'.format(model_name, model_version, end_time-start_time))
	return jsonify(response), 200
