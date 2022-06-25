'''
Final JSModel

requires following 6 files 
to be in the working dir
to successfully run:

- classification_tfidf_model
- clustering_tfidf_model
- classification_model
- clustering_model
- classification_features.json
- clustering_features.json
'''

import json
import pickle
import re

classification_labels = {
	"0": "marketing",
	"1": "cdn",
	"2": "tag-manager",
	"3": "video",
	"4": "customer-success",
	"5": "utility",
	"6": "ads",
	"7": "analytics",
	"8": "hosting",
	"9": "content",
	"10": "social",
	"11": "other"
}

clustering_labels = {
	"1": "noncritical",
	"0": "critical"
}

class JSModel(object):
	"""docstring for JSModel"""
	def __init__(self):
		with open("classification_tfidf_model", "rb") as f: model = f.read()
		self.tfidf1 = pickle.loads(model)

		with open("clustering_tfidf_model", "rb") as f: model = f.read()
		self.tfidf2 = pickle.loads(model)

		with open("classification_model", "rb") as f: model = f.read()
		self.classification_model = pickle.loads(model)

		with open("clustering_model", "rb") as f: model = f.read()
		self.clustering_model = pickle.loads(model)

		with open("classification_features.json") as f:
			self.classification_features = json.loads(f.read())["features"]

		with open("clustering_features.json") as f:
			self.clustering_features = json.loads(f.read())["features"]

		self.classification_kws = []
		for feature in self.classification_features:
			tmp = feature.split("|")
			self.classification_kws += tmp
		self.classification_kws = list(set(self.classification_kws))

		self.clustering_kws = []
		for feature in self.clustering_features:
			tmp = feature.split("|")
			self.clustering_kws += tmp
		self.clustering_kws = list(set(self.clustering_kws))

	def get_scripts_features(self, data, kws, features):
		resultant_features = []
		scripts_kws = []
		for kw in kws:
			scripts_kws += [kw]*data.count("."+kw+"(")
			scripts_kws += [kw]*data.count("."+kw+" (")
		for ft in features:
			if "|" not in ft:
				resultant_features += [ft]*scripts_kws.count(ft)
			else:
				singular_kws = ft.split("|")
				if len([ele for ele in singular_kws if ele in scripts_kws]) == len(singular_kws):
					resultant_features += [ft]
		return resultant_features

	def get_scripts_classification_features(self, data):
		return " ".join(self.get_scripts_features(data, self.classification_kws, self.classification_features))

	def get_scripts_clustering_features(self, data):
		return " ".join(self.get_scripts_features(data, self.clustering_kws, self.clustering_features))

	def printt(self):
		print(self.tfidf1)
		print(self.tfidf2)
		print(self.classification_model)
		print(self.clustering_model)

	def predict(self, script):
		script = re.sub("\s+", " ", script)
		reduced_script = self.get_scripts_classification_features(script)
		tfidf_representation = self.tfidf1.transform([reduced_script]).toarray()
		prediction = self.classification_model.predict(tfidf_representation)[0]
		max_prob = max(self.classification_model.predict_proba(tfidf_representation)[0])
		if max_prob >= 0.8:
			return classification_labels[str(prediction)]
		else:
			reduced_script = self.get_scripts_clustering_features(script)
			tfidf_representation = self.tfidf2.transform([reduced_script]).toarray()
			prediction = self.clustering_model.predict(tfidf_representation)[0]
			return clustering_labels[str(prediction)]