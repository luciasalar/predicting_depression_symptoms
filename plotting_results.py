import matplotlib.pyplot as plt
from model_training import *

os.environ['OT_QPA_PLATFORM'] = 'offscreen'




class plotting_results():
	'''plotting feature importance '''

	def __init__(self):
		self.grid_search = grid_search
		self.path = path 
		self.other_fea_names = other_fea_names
		
	def get_feature_names(self):
		'''combine tfidf features with other features'''
		tfidf_names = self.grid_search.best_estimator_.named_steps['feats'].transformer_list[0][1].steps[1][1].get_feature_names()
		names = tfidf_names + self.other_fea_names
		
		return names



	def get_feature_importance(self):
		'''plot feature importance and store results '''
		names = self.get_feature_names()

		importance = grid_search.best_estimator_.named_steps['clf'].steps[1][1].feature_importances_
		feat_importances = pd.Series(importance, index= names)
		feat_importances.nlargest(50).plot(kind='barh')
		plt.savefig(self.path + 'results/plots/feature_importance.png')
		plt.show()

		fea_df = pd.DataFrame(feat_importances)
		fea_df['features'] = fea_df.index
		fea_df.columns = ['importance','features']
		fea_df.to_csv(self.path + 'results/feature_importance/topFea_test.csv')


path = prepare.path
other_fea_names = training.get_other_feature_names(features_list)
plot = plotting_results()
plot.get_feature_importance()
