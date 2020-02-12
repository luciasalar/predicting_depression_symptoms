import numpy as np
from hmmlearn import hmm
from construct_mood_transition_feature import *
import pandas as pd
from numpy import array
import csv
import os.path
from sklearn.metrics import confusion_matrix, f1_score, precision_score,\
recall_score, confusion_matrix, classification_report, accuracy_score 
from collections import defaultdict

'''What you might consider is running multiple HMMs with different numbers of parameters (and maybe different "structures") and trying to find the "most parsimonious" model'''

class HMMtraining:
    def __init__(self, moodTransition, path, cutoff, ValenceObject):
        self.moodTransition = moodTransition
        self.path = path
        self.cutoff = cutoff
        self.ValenceObject = ValenceObject

    def get_train_test_id(self):
        '''get userid for low and high group '''
        participants = pd.read_csv(moodOb.path + 'participants_matched.csv')
        high_train = participants.loc[participants['cesd_sum'] > self.cutoff, ]
        low_train = participants.loc[participants['cesd_sum'] <= self.cutoff, ]
        return high_train['userid'].to_numpy(), low_train['userid'].to_numpy()

    def divide_users(self):
        high_train, low_train = self. get_train_test_id()

        high_train_dict = {}
        low_train_dict = {}

        for k, v in self.moodTransition.items():
            if k in high_train:
                high_train_dict[k] = v

            if k in low_train:
                low_train_dict[k] = v

        return high_train_dict, low_train_dict,

    def split_train_test(self):
        """split train test according to ML model"""

        read_train = pd.read_csv(self.path + 'train_feature.csv')
        read_test = pd.read_csv(self.path + 'test_feature.csv')
        train_id = read_train[['userid']]
        test_id = read_test[['userid']]

        train = {}
        test = {}
        for k, v in self.ValenceObject.items():
            if k in train_id.userid.values:
                train[k] = v
            if k in test_id.userid.values:
                test[k] = v
        return train, test


    def get_transition_m(self, dict_data):

         #transition states  
        posTOPos = 0 
        PosTNeg = 0
        PosTNeu = 0 
        PosTSil = 0

        negaToNega = 0
        NegTPos = 0
        NegTNeu = 0
        NegTSil = 0

        neuTNeu = 0
        NeuTPos = 0
        NeuTNeg = 0
        NeuSil = 0
        

        silToSil = 0
        silTNeg = 0
        silTNeu = 0
        silTPos = 0

        transMat = []
        positive_tran = []
        negative_tran =[]
        neutral_tran =[]
        silence_tran=[]

        transMat_pos = []
        transMat_neg =[]
        transMat_neu =[]
        transMat_sil =[]

        for k, v in dict_data.items():
            for transition in v: 
                if transition == 2:
                    posTOPos = posTOPos + 1
                if transition == 3:
                    PosTNeg = PosTNeg + 1
                if transition == 6:
                    PosTNeu = PosTNeu + 1
                if transition == 10:
                    PosTSil = PosTSil+ 1

                if transition == 1:
                    negaToNega = negaToNega + 1
                if transition == 4:
                    NegTPos = NegTPos + 1
                if transition == 8:
                    NegTNeu = NegTNeu+ 1
                if transition == 12:
                    NegTSil = NegTSil + 1

                if transition == 0:
                    neuTNeu = neuTNeu + 1
                if transition == 5:
                    NeuTPos = NeuTPos + 1
                if transition == 7:
                    NeuTNeg = NeuTNeg + 1
                if transition == 14:
                    NeuSil = NeuSil + 1

                if transition == 15:
                    silToSil = silToSil + 1
                if transition == 11:
                    silTNeg = silTNeg + 1
                if transition == 13:
                    silTNeu = silTNeu + 1
                if transition == 9:
                    silTPos = silTPos + 1

        positive_tran.append(posTOPos)
        positive_tran.append(PosTNeg)
        positive_tran.append(PosTNeu)
        positive_tran.append(PosTSil)

        negative_tran.append(negaToNega)
        negative_tran.append(NegTPos)
        negative_tran.append(NegTNeu)
        negative_tran.append(NegTSil)

        neutral_tran.append(neuTNeu)
        neutral_tran.append(NeuTPos)
        neutral_tran.append(NeuTNeg)
        neutral_tran.append(NeuSil)

        silence_tran.append(silToSil)
        silence_tran.append(silTNeg)
        silence_tran.append(silTNeu)
        silence_tran.append(silTPos)

        if sum(positive_tran) != 0:
            transMat_pos.append(posTOPos/sum(positive_tran))
            transMat_pos.append(PosTNeg/sum(positive_tran))
            transMat_pos.append(PosTNeu/sum(positive_tran))
            transMat_pos.append(PosTSil/sum(positive_tran))
        else: 
            transMat_pos.append([0,0,0,0])

        if sum(negative_tran) != 0:
            transMat_neg.append(negaToNega/sum(negative_tran))
            transMat_neg.append(NegTPos/sum(negative_tran))
            transMat_neg.append(NegTNeu/sum(negative_tran))
            transMat_neg.append(NegTSil/sum(negative_tran))
        else: 
            transMat_neg.append([0,0,0,0])

        if sum(neutral_tran) != 0:
            transMat_neu.append(neuTNeu/sum(neutral_tran))
            transMat_neu.append(NeuTPos/sum(neutral_tran))
            transMat_neu.append(NeuTNeg/sum(neutral_tran))
            transMat_neu.append(NeuSil/sum(neutral_tran))
        else: 
            transMat_neu.append([0,0,0,0])

        if sum(silence_tran) != 0:
            transMat_sil.append(silToSil/sum(silence_tran))
            transMat_sil.append(silTNeg/sum(silence_tran))
            transMat_sil.append(silTNeu/sum(silence_tran))
            transMat_sil.append(silTPos/sum(silence_tran))
        else: 
            transMat_sil.append([0,0,0,0])

        transMat.append(transMat_pos)
        transMat.append(transMat_neg)
        transMat.append(transMat_neu)
        transMat.append(transMat_sil)

        return transMat

    def get_fit_data(self, dict_data):
        '''return data for fitting'''
        fit_data = []
        for k, v in dict_data.items():
            fit_data.append(array(v))

        return fit_data, len(fit_data)-1

    def modelling(self, training_data):
        '''HMM model
        states: depressed, non-depressed, observations: sentiment (4 states)
        '''
        states = ['high_symptom', 'low_symptom']
        n_states = len(states)

        observations = ['positive', 'negative' , 'neutral', 'silence']
        n_observations = len(observations)

        #initialize transition prob 
        #     high  low
        #high 0.6,  0.4
        #low  0.4,  0.6

        transition_probability = np.array([
          [0.5, 0.5],
          [0.5, 0.5],
        ])

        #initialize emmission prob

                #positive negative neutral    silence
        #high     0.2      0.3      0.1          0.4
        #low      0.3      0.1      0.2          0.4

        emission_probability = np.array([
          [0.2, 0.3, 0,2, 0.3],
          [0.2, 0.2, 0.3, 0.3],
        ])

        #high low
        start_probability = np.array([0.3, 0.7]) #the latent variable

        model = hmm.MultinomialHMM(n_components=n_states, algorithm='viterbi', n_iter = 10, init_params = 'ste', random_state = 300)

        #initialization
        model.startprob_=start_probability
        model.transmat_=transition_probability
        model.emissionprob_=emission_probability

        #get training data and concatenate data
        train_data, lengths = H.get_fit_data(training_data)
        new = np.concatenate([[i for i in train_data]]).reshape(1,-1).T
        lengths = [len(i) for i in train_data]

        #label has to be integer
        new = np.array(new, dtype=np.float64)
        new[np.isnan(new)]= 3
        new[new == -1]= 2
        new = new.astype(np.int)

        #fit model
        model = model.fit(new, lengths)

        return model, transition_probability, emission_probability, start_probability

    def write_results(self, training_data, datasetName):
        model, transmat_init, emission_init, start_probability = self.modelling(training_data)

        file_exists = os.path.isfile(self.path + 'results/HMM_result.csv')
        f = open(self.path + 'results/HMM_result.csv' , 'a') 
        writer_top = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer_top.writerow(['emissionprob', 'transitionprob', 'training_data', 'initialized_transprob', 'initialized_emission_prob', 'start_probability'])

        result_row = [[model.emissionprob_, model.transmat_, datasetName, transmat_init, emission_init, start_probability]]
        writer_top.writerows(result_row)

        f.close()

        return model 


class Prediction:
    def __init__(self, train, test, train_model, n_days, timewindow, path):
        self.train = train
        self.test = test
        self.train_model = train_model
        self.n_days = n_days
        self.timewindow = timewindow
        self.path = path

    def recode(self):
        recode_dict = {}
        for k, v in self.test.items():
            new_l = []
            for i in v:
                if i == -1:
                    new_l.append(2)
                elif np.isnan(i) == True:
                    new_l.append(3)
                else:
                    new_l.append(1)
            recode_dict[k] = new_l
        return recode_dict

    def predict(self):
        model, transition_probability, emission_probability, start_probability = self.train_model.modelling(self.train)
        test = self.recode()
        result_dict = {}
        result_prob = {}
        for k, v in test.items():
            result = model.predict(np.asarray(v).reshape(-1, 1))
            result_proba = model.predict_proba(np.asarray(v).reshape(-1, 1))
            result_dict[k] = result
            result_prob[k] = result_proba[1][1] #get prob of class1(positive)
        result_df = pd.DataFrame.from_dict(result_dict).T
        result_df.to_csv(self.path + 'results/hmm_recover.csv')

        return result_dict, result_prob


    def classify(self):
        result_dict, result_prob = self.predict()
        hmm_classify ={}
        for k, v in result_dict.items():
            selected_date = v[-self.timewindow:]
            count = 0
            for i in selected_date:
                if i == 1:
                    count = count + 1 
            if count >= self.n_days:
                hmm_classify[k] = 1
            else:
                hmm_classify[k] = 0
        
        hmm_df = pd.DataFrame.from_dict(hmm_classify, orient='index')
        hmm_df.to_csv(self.path + 'results/result_recover.csv')
        return hmm_classify

    def hmm_report(self):
        cesd_sum = pd.read_csv(self.path + 'participants_matched.csv')
        cesd_sum = cesd_sum[['userid','cesd_sum']]
        # merge results
        HMM_result = pd.read_csv(self.path + 'results/result_recover.csv')
        HMM_result.columns = ['userid','hmm_pre']
        all_data = cesd_sum.merge(HMM_result, on='userid')
        all_data.loc[all_data['cesd_sum'] > 22, 'cesd'] = 1
        all_data.loc[all_data['cesd_sum'] <= 22, 'cesd'] = 0
        report = classification_report(all_data['cesd'], all_data['hmm_pre'], output_dict=True)
        return report


 



    # def analysis


#get mood dictionary
moodOb = MoodFeature(path = path, participants = participants)
ValenceObject = moodOb.get_mood_vector(365) #user data in the past 365 days
path = moodOb.path

##get mood transition matrix, this is not used in the model but very useful for providing information 
transition = TransitionMatrix(ValenceObject = ValenceObject, windowSize=14) 
#get transition probabilities between sentiment states
moodTransition = transition.get_mood_transitions()

#model training 
H = HMMtraining(moodTransition = moodTransition, path = path, cutoff=22, ValenceObject= ValenceObject)

#get mood transition from low and high group 
high_train_dict, low_train_dict= H.divide_users()
High_transition_probability = H.get_transition_m(high_train_dict)

#get HMM model
model = H.write_results(ValenceObject, 'frequent_users')
train, test = H.split_train_test()

#train one model on high symptom group and one model on low symptom group?
pre  = Prediction(train=train, test=test, train_model=H, n_days=3, timewindow=7, path=path)
# recode = pre.recode()
hmm_re = pre.hmm_report()
