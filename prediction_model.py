import pandas as pd
from imblearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import confusion_matrix, f1_score, precision_score,\
recall_score, confusion_matrix, classification_report, accuracy_score 
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support
import csv
import numpy as np
from sklearn.model_selection import LeaveOneOut
import pickle
import datetime
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from inspect import signature

path = '/home/lucia/phd_work/mypersonality_data/predicting_depression_symptoms/data/'

def prepare_data(path, moodFeatureFile, transitionFeatureFile):
	'''merging data'''
	mood_feature = pd.read_csv(path + moodFeatureFile)
	mood_feature  = mood_feature.rename(columns = {"Unnamed: 0":"userid"}) 
	participants = pd.read_csv(path + 'participants_matched.csv')
	participants  = participants[['userid','cesd_sum']]
	mood_transitions = pd.read_csv(path + transitionFeatureFile)
	mood_transitions.rename(columns={'Unnamed: 0':'userid'}, inplace=True)
	mood_feature2 = pd.merge(mood_feature, mood_transitions, on = 'userid')
	feature_cesd = pd.merge(mood_feature2, participants, on = 'userid')
	return feature_cesd
	#return mood_transitions

#f = prepare_data(path, 'mood_vectors/mood_vector_frequent_user_window3.csv', 'mood_vectors/mood_transition_one_hoc_frequent_user_window_3.csv')



def get_y(feature_df):
	'''get y '''
	y = feature_df['cesd_sum']
	return y

def recode_y(y, threshold):
	'''recode y to binary according to a threshold'''
	new_labels = []
	for i in y:
		if i <= threshold:
			i = 0
		if i > threshold:
			i = 1
		new_labels.append(i)
	return new_labels


def get_feature(feature_df):
	'''get feature matrix'''
	features = feature_df.iloc[:,1:feature_df.shape[1]-1]
	return features

def get_train_test_split(X, y):
    '''split train test'''
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 300)
    return X_train, X_test, y_train, y_test

def RF_classifier(X_train, y_train, y_test, X_test): 
    '''train svm'''
    cv_fold = StratifiedKFold(n_splits=10, random_state=0)
    rf = make_pipeline(RandomForestClassifier())
    parameters = [{'randomforestclassifier__max_features': ['auto','sqrt','log2'], 
                   'randomforestclassifier__max_leaf_nodes': [500,1000,1500, 2000],
                   'randomforestclassifier__max_depth':[5,10,15],
                   'randomforestclassifier__n_estimators':[50,100,300,500]}]

    grid_search_item = GridSearchCV(rf,
                                    param_grid=parameters,
                                    cv=cv_fold,
                                    scoring='accuracy',
                                    n_jobs=-1)

    grid_search = grid_search_item.fit(X_train, y_train)
    y_true, y_pred = y_test, grid_search.predict(X_test)
    return y_true, y_pred, grid_search


def SVM_classifier(X_train, y_train, y_test, X_test): 
    '''train svm'''

    cv_fold = StratifiedKFold(n_splits=3, random_state=0)
    svc = make_pipeline(svm.SVC())
    parameters = [{'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
                   'svc__gamma': [0.5, 0.1, 0.01, 0.001, 0.0001],
                   'svc__C':[0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0, 10],
                   'svc__class_weight':['balanced']}]

    grid_search_item = GridSearchCV(svc,
                                    param_grid=parameters,
                                    cv=cv_fold,
                                    scoring='accuracy',
                                    n_jobs=-1)
    grid_search = grid_search_item.fit(X_train, y_train)
    y_true, y_pred = y_test, grid_search.predict(X_test)
    return y_true, y_pred, grid_search


def plot_precision_recall(y_true, y_pred, figName, path):
	'''plot precision and recall'''
	average_precision = average_precision_score(y_true, y_pred)
	precision, recall, _ = precision_recall_curve(y_true, y_pred)

	# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
	step_kwargs = ({'step': 'post'}
	               if 'step' in signature(plt.fill_between).parameters
	               else {})
	plt.step(recall, precision, color='b', alpha=0.2,
	         where='post')
	plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
	          average_precision))
	plt.savefig(path + 'plots/plot_{}_{}.pdf'.format(figName,str(datetime.datetime.now())))


def store_results(path, y_true, y_pred, grid_search, y_label_name, moodFeatureFile):
    '''save prediciton result to file'''
    report = precision_recall_fscore_support(y_true, y_pred)
    precision, recall, fscore, support=precision_recall_fscore_support(y_true, y_pred, average='macro')

    y_pred_series = pd.DataFrame(y_pred)
    y_true_series = pd.DataFrame(y_true)
    result = pd.concat([y_true_series, y_pred_series], axis=1)
    result.columns = ['y_true', 'y_pred']
    result.to_csv(path + 'prediction_result_{}_{}.csv'.format(grid_search.estimator.steps[0][0], str(datetime.datetime.now())))
    plot_precision_recall(y_true, y_pred, grid_search.estimator.steps[0][0], path)

    #precision and recall on test set 
    scores = precision_recall_fscore_support(y_true, y_pred, average='macro')

    f = open(path +'model_results.csv', 'a')
    writer_top = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
    writer_top.writerow(['classifer'] + ['best_scores'] + ['best_parameters'] + ['classification_report_GridSearch'] + ['marco_precision']+['marco_recall']+['marco_fscore']+['support']+['runtime']+['vectorType'] + ['y_label'] + ['feature_file'])
    result_row = [[grid_search.estimator.steps[0][0], grid_search.best_score_, grid_search.best_params_, report, scores[0], scores[1], scores[2], scores[3], str(datetime.datetime.now()), path.split('/')[-1], y_label_name, moodFeatureFile]]
    writer_top.writerows(result_row)
    f.close

def select_features(all_features, featureName):
    #all_feature = prepare_data(path,  moodFeatureFile, transitionFeatureFile)
    if featureName == 'transitions':
        selected = [col for col in all_features.columns if 'transitions' in col]
        selected_c = all_features[selected]
        return selected_c

    elif featureName == 'mood':
        selected = [col for col in all_features.columns if 'mood' in col]
        selected_c = all_features[selected]
        return selected_c

	


def run_model(path, moodFeatureFile, transitionFeatureFile): 
    '''run model, plot and store results'''

    all_features = prepare_data(path,  moodFeatureFile, transitionFeatureFile)
    y_cesd = get_y(all_features)
   # X = get_feature(all_features)
    y_recode = recode_y(y_cesd, 22) #recode y to 1, 0 according to threshold
    X = select_features(all_features, 'mood')

    X_train, X_test, y_train, y_test = get_train_test_split(X, y_recode)
    y_label_name = y_cesd.name

    #prediction and store files
    y_true, y_pred, grid_search = RF_classifier(X_train, y_train, y_test, X_test)
    store_results(path + 'results/', y_true, y_pred, grid_search, y_label_name, moodFeatureFile)

    y_true, y_pred, grid_search = SVM_classifier(X_train, y_train, y_test, X_test)
    store_results(path + 'results/', y_true, y_pred, grid_search, y_label_name, moodFeatureFile)
   
#merge with CESD

all_features = run_model(path, 'mood_vectors/mood_vector_frequent_user_window_3.csv', 'mood_vectors/mood_transition_one_hoc_frequent_user_window_3.csv')

# def loop_the_grid(path_to_psy, path_to_valencefile, path_to_save, path_to_valFreq, days_for_model, feature_name):
#     #loop days
#     for day in days:
#       run_model(path_to_psy, path_to_valencefile, path_to_save, path_to_valFreq, day, 'valenceVec')

