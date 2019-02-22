import os
from importlib import import_module
from flask import Flask, g, request, jsonify
from flask_pymongo import PyMongo
from utils.io import UTF8Encoder, IDtoString
# from werkzeug import find_modules
# from .auth.control import AuthControl


def register_blueprints(app):
    current_directory = os.path.abspath(os.path.dirname(__file__))
    modules = [name for name in os.listdir(current_directory) if os.path.isdir(os.path.join(current_directory, name)) and name != '__pycache__']
    for name in modules:
        try:
            mod = __import__(name, globals(), locals(), [], 1)
            app.register_blueprint(mod.blue_print)
        except ModuleNotFoundError as e:
            error_message = name + " module is missing or it lacks a blueprint."
            raise Exception(error_message)
    return None

def register_middleware(app, mongo_main, mongo_ai):
    @app.before_request
    def before_request():
        # mongo_main.db.add_son_manipulator(UTF8Encoder())
        # mongo_main.db.add_son_manipulator(IDtoString())
        # mongo_main.db.add_son_manipulator(StringtoID())
        g.db_main = mongo_main.db

        # mongo_ai.db.add_son_manipulator(UTF8Encoder())
        # mongo_ai.db.add_son_manipulator(IDtoString())
        # mongo_ai.db.add_son_manipulator(StringtoID())
        g.db_ai = mongo_ai.db

    # @app.before_request
    # def check_token():
    #     auth_token = request.headers.get('Authorization')
    #     if auth_token is not None:
    #         try:
    #             user_id = AuthControl.decode_auth_token(auth_token)
    #         except Exception as e:
    #             return jsonify({'message': e.__str__()}), 401
    #         g.user_id = user_id
    #     return None

    # ERROR HANDLER
    @app.errorhandler(404)
    def not_found(error):
        return "Page not found!", 404

def create_app(testing=False):
    app = Flask("foodstreet_nucleus")
    app.config.from_object('config')
    # if testing:
        # app.config.from_object('config_testing')

    # ESTABLISH & INJECT DB CONNECTION
    app.config["MONGO_URI"] = app.config["MONGO_URI_MAIN"]
    mongo_main = PyMongo(app)

    app.config["MONGO_URI"] = app.config["MONGO_URI_AI"]
    mongo_ai = PyMongo(app)

    register_blueprints(app)
    register_middleware(app, mongo_main, mongo_ai)
    return app

app = create_app()


