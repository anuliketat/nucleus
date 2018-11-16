class NoClass(Exception):
	def __init__(self, class_name):
		self.class_name = class_name

	def http_status(self): 
		return 400

	def __str__(self):
		return 'ERROR! Class: ' + self.class_name + ' does not exist! Please check the model and version!'

class NoModule(Exception):
	def __init__(self, module_name):
		self.module_name = module_name

	def http_status(self): 
		return 400

	def __str__(self):
		return 'ERROR! Module: ' + self.module_name + ' does not exist! Please check the model and version!'

# class NoUser(Exception):
# 	def __init__(self, user_id):
# 		self.user_id = user_id

# 	def http_status(self): 
# 		return 404

# 	def __str__(self):
# 		return 'ALERT! User with ID: ' + self.user_id + ' does not exist!'

class NoModel(Exception):
	def __init__(self, model_name, model_version):
		self.model_name = model_name
		self.model_version = model_version

	def http_status(self):
		return 426

	def __str__(self):
		return 'ALERT! The ML Model: ' + self.model_name + self.model_version + ' is not yet trained! Please update the model!'

class DBError(Exception):
	def __init__(self, db_name, error):
		self.db_name = db_name
		self.error = error

	def http_status(self):
		return 500

	def __str__(self):
		return 'ERROR! DB name: ' + self.db_name + ' and error message: ' + str(self.error)