import os
from pymongo.son_manipulator import SONManipulator
from flask import jsonify
from bson.objectid import ObjectId
from bson.binary import Binary

class UTF8Encoder(SONManipulator):
    def transform_outgoing(self, son, collection):
        new_son = {}
        for key, value in son.items():
            try:
                if isinstance(key, bytes):
                    key = key.decode('UTF-8')
                if isinstance(value, bytes):
                    value = value.decode('UTF-8')
            except Exception as e:
                print(e)
            new_son[key] = value
        son = new_son
        return son

    def will_copy(self):
        return False

class IDtoString(SONManipulator):
    def transform_outgoing(self, son, collection):
        new_son = {}
        for key, value in son.items():
            if isinstance(value, ObjectId):
                value = str(value)
            new_son[key] = value
        son = new_son
        return son

    def will_copy(self):
        return False
