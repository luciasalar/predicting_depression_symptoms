import pandas as pd
from sklearn.model_selection import GridSearchCV
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
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support
import csv
import numpy as np
from sklearn.model_selection import LeaveOneOut
import pickle
from moodVector.MoodVector import getTransitions, getUserTransitions
#from datetime import datetime
import datetime



def merge_cesd_valence(path_to_psy, valence_file):
    '''read and merge cesd and valence'''
    all_psy = pd.read_csv(path_to_psy + 'ValenceEmptyFreqAllVar.csv')
    selected_psy = all_psy[['userid', 'ope', 'con', 'ext', 'agr', 'neu', 'swl', 'CESD_sum']]
    val_psy = pd.merge(selected_psy, valence_file, how='left', on='userid')

    return val_psy


def recode(array):
    '''recode y, cesd > 23 as high depressive symptoms'''
    new = []
    for num in array:
        if num <= 23:
            new.append(0)
        if num > 23:
            new.append(1)
    return new

def recode_vector(valence_vector):
    '''values of the vector is categorical, we can treat them as ordinal, here we recode the value to make them oridnal
    according to the presences of positive affect

    negative = 1 -> -1   lack of positive affect
    mix = 3 -> 2
    positive = 2 -> 3
    neutral = 4 -> 1
    empty = -1 -> 0 no information
    '''
    new_df = valence_vector.replace([-1, 1, 2, 3, 4], [0, -1, 3, 2, 1])

    return new_df


def get_y(val_psy):
    '''split X, y here  y can be any psychological characteristics'''
    #X = val_psy.iloc[:, 8: 8+days_for_model]  #here you can customize the days
    y_cesd = val_psy["CESD_sum"]
    y_swl = val_psy["swl"]
    y_neu = val_psy["neu"]
    y_agr = val_psy["agr"]
    y_ext = val_psy["ext"]
    y_con = val_psy["con"]
    y_ope = val_psy["ope"]
    return y_cesd, y_swl, y_neu, y_agr, y_ext, y_con, y_ope

def get_train_test_split(X, y):
    '''split train test'''
    y_recode = recode(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_recode, test_size=0.30, random_state = 300)
    return X_train, X_test, y_train, y_test

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

def RF_classifier(X_train, y_train, y_test, X_test): 
    '''train svm'''
    cv_fold = StratifiedKFold(n_splits=10, random_state=0)
    rf = make_pipeline(RandomForestClassifier())
    parameters = [{'randomforestclassifier__max_features': ['auto','sqrt','log2'], 
                   'randomforestclassifier__max_leaf_nodes': [50,100,300,500,1000],
                   'randomforestclassifier__max_depth':[5,10,15,20],
                   'randomforestclassifier__n_estimators':[50,100,300,500]}]

    grid_search_item = GridSearchCV(rf,
                                    param_grid=parameters,
                                    cv=cv_fold,
                                    scoring='accuracy',
                                    n_jobs=-1)
    grid_search = grid_search_item.fit(X_train, y_train)
    y_true, y_pred = y_test, grid_search.predict(X_test)
    return y_true, y_pred, grid_search



def store_results(path_to_save, y_true, y_pred, grid_search, days_for_model, path_to_valencefile, y_label_name, feature_name):
    '''save prediciton result to file'''
    report = precision_recall_fscore_support(y_true, y_pred)
    precision, recall, fscore, support=precision_recall_fscore_support(y_true, y_pred, average='macro')

    y_pred_series = pd.DataFrame(y_pred)
    y_true_series = pd.DataFrame(y_true)
    result = pd.concat([y_true_series, y_pred_series], axis=1)
    result.columns = ['y_true', 'y_pred']
    result.to_csv(path_to_save + 'best_result.csv')

    f = open(path_to_save +'result.csv', 'a')
    writer_top = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
    writer_top.writerow(['classifer'] + ['best_scores'] + ['best_parameters'] + ['classification_report'] + ['marco_precision']+['marco_recall']+['marco_fscore']+['support']+['runtime']+['affect_days']+['vectorType'] + ['y_label'] + ['feature_set'])
    result_row = [[grid_search.estimator.steps[0][0], grid_search.best_score_, grid_search.best_params_, report, precision, recall, fscore, support, str(datetime.datetime.now()), days_for_model, path_to_valencefile.split('/')[-1], y_label_name], feature_name]
    writer_top.writerows(result_row)
    f.close

def read_valence_vector(path_to_valencefile):
    '''convert vector pickle to df'''
    moodVec = pickle.load(open(path_to_valencefile, "rb" ))
    moodVec_df = pd.DataFrame.from_dict(moodVec, orient='index').reset_index().rename(columns={'index':'userid'})
    moodVec_df_recode = recode_vector(moodVec_df)
    return moodVec_df_recode

def get_new_vector(path_to_valencefile, days_for_model):
    moodVec = pickle.load(open(path_to_valencefile, "rb" ))
    selected_vec = {}
    for k, v in moodVec.items():
        selected_vec[k] = v[0:days_for_model]
    return selected_vec

def compute_transition_prob(TransitionStates):
    selected_vec_df = pd.DataFrame.from_dict(TransitionStates, orient='index').reset_index()
    selected_vec_df.columns = ['userid','emptyTran', 'negaTran', 'posiTran', 'mixTran', 'neuTran', 'PosAndNeg', 'MixAndPos', 'MixAndNeg', 'MixAndNeu', 'NeuAndPos', 'NeuAndNeg', 'EmptyAndPos', 'EmptyAndNeg', 'EmptyAndMix', 'EmptyAndNeu']
    selected_vec_df['allPosts'] = selected_vec_df.sum(axis=1) 

    #we compute the pobability by dividing the transition with days: 59
    selected_vec_df.index = selected_vec_df['userid']
    selected_vec_df = selected_vec_df.drop(['userid'], axis=1)
    Tranprob = selected_vec_df.apply(lambda x: x/59)

    return Tranprob

def get_transition_counts(path_to_valencefile):
    selected_vec = get_new_vector(path_to_valencefile, 30)
    TransitionStates = getUserTransitions(selected_vec)
    TransitionStates_prob = compute_transition_prob(TransitionStates)
    return TransitionStates_prob


def prepare_data(path_to_psy, path_to_valencefile, path_to_valFreq):
    '''merge all the features and return'''
    valence_file = read_valence_vector(path_to_valencefile)
    all_cases = merge_cesd_valence(path_to_psy, valence_file)
    #valFreq = pd.read_csv(path_to_valFreq)
    transition_state = get_transition_counts(path_to_valencefile)
    #valFreq_fea = valFreq[['emptyTran', 'negaTran', 'posiTran', 'mixTran', 'neuTran',
    #   'PosAndNeg', 'MixAndPos', 'MixAndNeg', 'MixAndNeu', 'NeuAndPos',
    #   'NeuAndNeg', 'EmptyAndPos', 'EmptyAndNeg', 'EmptyAndMix', 'EmptyAndNeu', 'userid']]
    all_features = pd.merge(all_cases, transition_state, how='left', on='userid')

    return all_features


def select_features(all_features, feature_name, days_for_model):
    '''select feature for prediction model'''
    if feature_name == 'valenceVec':
       selected_features = all_features.iloc[:, 8:days_for_model+8]
    elif feature_name == 'transition_states':
       selected_features = all_features.iloc[:, 69:all_features.shape[1]]
    elif feature_name == 'valenceVec_tran':
       selected_features = pd.concat([all_features.iloc[:, 8:days_for_model+8], all_features.iloc[:, 69:all_features.shape[1]]], axis = 1)
    return selected_features


def run_model(path_to_psy, path_to_valencefile, path_to_save, path_to_valFreq, days_for_model, feature_name): 
    #valence_file = read_valence_vector(path_to_valencefile)
    #all_cases = merge_cesd_valence(path_to_psy, valence_file)
    all_features = prepare_data(path_to_psy, path_to_valencefile, path_to_valFreq)
    y_cesd, y_swl, y_neu, y_agr, y_ext, y_con, y_ope = get_y(all_features)
    X = select_features(all_features, feature_name, days_for_model)
    #X = select_features(all_features, 'valenceVec', 60)

    X_train, X_test, y_train, y_test = get_train_test_split(X, y_cesd)
    y_label_name = y_cesd.name
    #y_true, y_pred, grid_search = SVM_classifier(X_train, y_train, y_test, X_test)
    #store_results(path_to_save + 'results/', y_true, y_pred, grid_search, days_for_model, path_to_valencefile, y_label_name, feature_name)
    y_true, y_pred, grid_search = RF_classifier(X_train, y_train, y_test, X_test)
    store_results(path_to_save + 'results/', y_true, y_pred, grid_search, days_for_model, path_to_valencefile, y_label_name, feature_name)


#merge with CESD
path_to_valencefile = '/home/lucia/phd_work/mypersonality_data/cognitive_distortion/newScripts/moodVector/moodVectorsData/MoodVecDes1.pickle'
path_to_psy = '/home/lucia/phd_work/mypersonality_data/cognitive_distortion/newScripts/moodVector/moodVectorsData/'
path_to_save = '/home/lucia/phd_work/mypersonality_data/cognitive_distortion/data/important_data/'
path_to_valFreq = '/home/lucia/phd_work/mypersonality_data/cognitive_distortion/newScripts/moodVector/moodVectorsData/ValenceEmptyAllVar.csv'
run_model(path_to_psy, path_to_valencefile, path_to_save, path_to_valFreq, 60, 'valenceVec')

days = [10, 20, 30, 40, 50, 60]
def loop_the_grid(path_to_psy, path_to_valencefile, path_to_save, path_to_valFreq, days_for_model, feature_name):
    #loop days
    for day in days:
      run_model(path_to_psy, path_to_valencefile, path_to_save, path_to_valFreq, day, 'valenceVec')






#see feature importance
#add compute transition states

