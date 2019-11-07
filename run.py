from model_training import *
import csv
import sklearn 
import datetime
import json
import pickle


def loop_experiment(main_dir, experiment, X_train, y_train, X_test, y_test):
	f = open(main_dir + 'results/result.csv' , 'w')
	writer_top = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
	writer_top.writerow(['best_scores'] + ['best_parameters'] + ['marco_precision']+['marco_recall']+['marco_fscore']+['support'] +['time'] + ['model'] +['feature_set'])

	for classifer in experiment['experiment']:
		for feature in experiment['features']:

			

			#pipeline = set_pipeline(experiment['features'][feature], eval(classifer)())
			#parameters = experiment['experiment'][classifer]
			precision, recall, fscore, support,grid_search = c.test_model(p.path, classifer, feature)
			
			#grid_search = training_models(pipeline, parameters, X_train, y_train)


			#report, precision, recall, fscore, support = test_model(X_test, y_test, main_dir, classifer, grid_search)

			result_row = [[grid_search.best_score_, grid_search.best_params_, precision,recall,fscore,support, str(datetime.datetime.now()), classifer, feature]]


			# if classifer == 'sklearn.ensemble.RandomForestClassifier':
				

			#  	importances, std_importances, indices =  calculate_feature_importance_rf(grid_search)

			#  	plot_feature_importance_rf(grid_search, X_train, importances, std_importances, indices, (16,4))
			 	#get_feature_importance(grid_search, main_dir, X_train)

			writer_top.writerows(result_row)
			f.close
	f.close
	return grid_search

# class TrainingClassifiers(): 
# 	def __init__(self):
# 		self.X_train = X_train
# 		self.y_train = y_train
# 		self.X_test = X_test
# 		self.y_test= y_test
		




loop_experiment(p.path, experiment, X_train, y_train, X_test, y_test)


# def loop_experiment(main_dir, experiment, X_train, y_train, X_test, y_test):
# 	f = open(main_dir + str(datetime.datetime.now()) +'result.csv' , 'w')
# 	writer_top = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
# 	writer_top.writerow(['best_scores'] + ['best_parameters'] + ['classification_report'] + ['marco_precision']+['marco_recall']+['marco_fscore']+['support'])

# 	for classifer in experiment['experiment']:
# 		for feature in experiment['features']:

# 			logging.info('[{}] setting up pipeline: {}'.format(str(datetime.datetime.now()),classifer))
# 			pipeline = set_pipeline(experiment['features'][feature], eval(classifer)())
# 			parameters = experiment['experiment'][classifer]

# 			logging.info('[{}] grid_search: {}'. format(str(datetime.datetime.now()), classifer))
# 			grid_search = training_models(pipeline, parameters, X_train, y_train)


# 			report, precision, recall, fscore, support = test_model(X_test, y_test, main_dir, classifer, grid_search)

# 			result_row = [[grid_search.best_score_, grid_search.best_params_, report, precision,recall,fscore,support]]


# 			if classifer == 'sklearn.ensemble.RandomForestClassifier':
				

# 			 	importances, std_importances, indices =  calculate_feature_importance_rf(grid_search)

# 			 	plot_feature_importance_rf(grid_search, X_train, importances, std_importances, indices, (16,4))
# 			 	#get_feature_importance(grid_search, main_dir, X_train)

# 			writer_top.writerows(result_row)
# 	f.close
# 	return grid_search

#if __name__ == '__main__':

main_dir = '/Users/lucia/phd_work/Clpsy/'
log_path = main_dir + '/logFiles/'
data_path = main_dir + '/data/clpsych19_training_data/'
feature_path = main_dir +'/suicideDetection/features/'

# reddit_post, labels = get_data(data_path)

# logging_time = str(datetime.datetime.now())
# logging.basicConfig(filename= log_path + logging_time + 'log_file.log',level=logging.DEBUG)

# all_features = read_features(feature_path, reddit_post)

# y = all_features.raw_label
# X_train, X_test, y_train, y_test = train_test_split(all_features, y, test_size=0.30, random_state=35)

# experiment = load_experiment(main_dir + '/suicideDetection/experiment/experiment.yaml')


# grid_search = loop_experiment(main_dir, experiment, X_train, y_train, X_test, y_test)


#sorted_predictors = [x for (y, x) in sorted(zip(importances, X_train.columns), reverse=True)]










