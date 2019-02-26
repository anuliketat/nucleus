import time
from flask import Blueprint, g, jsonify, request

from .control import MAControl
from api.exceptions import NoClass
from utils.misc import get_traceback, logger

blue_print = Blueprint('mancio', __name__, url_prefix='/mancio')

@blue_print.route('/update-model/<model_name>/<model_version>', methods=['GET'])
def update_model(model_name, model_version):
	start_time = time.time()

	try:
		forecast_engine = MAControl(g.db_main, g.db_ai, g.fs_ai, model_name, model_version)
		response = forecast_engine.update_model()
	except NoClass as e:
		logger('NUCLEUS_MANCIO', 'ERR', e.traceback())
		logger('NUCLEUS_MANCIO', 'ERR', e.__str__())
		return jsonify({'message': e.__str__()}), e.http_status()
	except Exception as e:
		logger('NUCLEUS_MANCIO', 'ERR', get_traceback(e))
		return jsonify({'message': 'Unknown error! Please try after sometime.'}), 500

	end_time = time.time()
	logger('NUCLEUS_MANCIO', 'EXE_TIME', 'Execution time of update_model() for {}_{} is {}.'.format(model_name, model_version, end_time-start_time))
	return jsonify(response), 200