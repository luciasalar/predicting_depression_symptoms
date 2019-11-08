import matplotlib.pyplot as plt


class plotting_results():

	def __init__(self):
		self.grid_search = grid_search
		self.path = path
		self.X_train = X_train


	def get_feature_importance(self, X_train):
		importance = self.grid_search.best_estimator_.named_steps['clf'].steps[1][1].feature_importances_
		print(X_train.shape)
		feat_importances = pd.Series(importance, index= self.X_train.columns)
		feat_importances.nlargest(20).plot(kind='barh')
		plt.show()

		fea_df = pd.DataFrame(feat_importances)
		fea_df['features'] = fea_df.index
		fea_df.columns = ['importance','features']
		fea_df.to_csv(self.path + 'results/topFea_test.csv')

	def calculate_feature_importance_rf():
	    importances = self.grid_search.best_estimator_.named_steps['clf'].steps[1][1].feature_importances_
	    std_importances = np.std([tree.feature_importances_ for tree in grid_search.best_estimator_.named_steps['clf'].steps[1][1].estimators_],
	            axis=0)
	    indices = np.argsort(importances)[::-1]
	    return importances, std_importances, indices

def plot_feature_importance_rf(matrix, mean_importance, std_importance, indices, fig_size):
    tfidf_names = grid_search.best_estimator_.named_steps['feats'].transformer_list[0][1].named_steps['cv'].get_feature_names()
    feature_names = pd.concat([pd.Series(tfidf_names), pd.Series(matrix.columns.tolist)])
    sorted_predictors = [x for (y, x) in sorted(zip(mean_importance, feature_names[:-1]), reverse=True)]

    fig, ax = plt.subplots(figsize=fig_size)
    
    plt.bar(range(len(indices)), mean_importance[indices],
            color="m", yerr=std_importance[indices], align="center")
    plt.xticks(range(len(indices)), sorted_predictors, rotation=90)
    plt.xlim([-1, len(indices)])
    ##£
    plt.show()
   
    return fig, ax
