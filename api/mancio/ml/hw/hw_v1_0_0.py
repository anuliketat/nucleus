from ..basic_model import basic_model

class hw_v1_0_0(basic_model):
    def __init__(self):
        self.model_name = 'hw'
        self.model_version = 'v1.0.0'

    def update_model(self, db_main, db_ai, fs_ai):
        print('update_model() function called for: {}_{}.'.format(self.model_name, self.model_version))