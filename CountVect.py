from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import *
from nltk.corpus import stopwords
import re
import numpy as np
from multiprocessing import Pool
import multiprocessing
import time



class Count_Vect(): 

    #path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/'

    def __init__(self):

        contractions_list = { 
        "ain't": "am not / are not / is not / has not / have not",
        "aren't": "are not / am not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he had / he would",
        "he'd've": "he would have",
        "he'll": "he shall / he will",
        "he'll've": "he shall have / he will have",
        "he's": "he has / he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how has / how is / how does",
        "I'd": "I had / I would",
        "I'd've": "I would have",
        "I'll": "I shall / I will",
        "I'll've": "I shall have / I will have",
        "I'm": "I am",
        "I've": "I have",
        "isn't": "is not",
        "it'd": "it had / it would",
        "it'd've": "it would have",
        "it'll": "it shall / it will",
        "it'll've": "it shall have / it will have",
        "it's": "it has / it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she had / she would",
        "she'd've": "she would have",
        "she'll": "she shall / she will",
        "she'll've": "she shall have / she will have",
        "she's": "she has / she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as / so is",
        "that'd": "that would / that had",
        "that'd've": "that would have",
        "that's": "that has / that is",
        "there'd": "there had / there would",
        "there'd've": "there would have",
        "there's": "there has / there is",
        "they'd": "they had / they would",
        "they'd've": "they would have",
        "they'll": "they shall / they will",
        "they'll've": "they shall have / they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we had / we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what shall / what will",
        "what'll've": "what shall have / what will have",
        "what're": "what are",
        "what's": "what has / what is",
        "what've": "what have",
        "when's": "when has / when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where has / where is",
        "where've": "where have",
        "who'll": "who shall / who will",
        "who'll've": "who shall have / who will have",
        "who's": "who has / who is",
        "who've": "who have",
        "why's": "why has / why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had / you would",
        "you'd've": "you would have",
        "you'll": "you shall / you will",
        "you'll've": "you shall have / you will have",
        "you're": "you are",
        "you've": "you have"
        }

        self.contractions = contractions_list
        self.num_partitions = 15
        self.num_cores = 10
        self.path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/'
        #self.path = path 

    def preprocess(self, sent):

        words = str(sent).split()
        new_words = []
        ps = PorterStemmer()
        for w in words:
            if w in list(self.contractions.keys()):
                w = self.contractions[w]
            # remove non English word characters
            w = re.sub(r'[^\x00-\x7F]+',' ', w).lower()
            # remove puncutation 
            w = re.sub(r'[^\w\s]','',w)
            #remove digits
            w = re.sub(r'[0-9]+', '', w)  
            w = ps.stem(w)
            
                
            if w not in set(stopwords.words('english')):
                new_words.append(w)
            
        return ' '.join(new_words)


    # def prepare_text_data(self):
    #     '''merging data, select frequent users, return df with text and user id'''
    #    #select frequent users, here you need to change the matched user files if selection criteria changed
    #     participants = pd.read_csv(self.path + 'participants_matched.csv')   
    #     text = pd.read_csv(self.path + 'status_sentiment.csv')
    #     text_merge = pd.merge(participants, text, on = 'userid')
    #     text_fea = text_merge[['userid','text']]
    #     text_fea['text'] = text_fea.groupby(['userid'])['text'].transform(lambda x: ','.join(x))
    #     text_fea['userid'].drop_duplicates()
        
    #     return text_fea

    def prepare_text_data(self):
        '''merging data, select frequent users, return df with text and user id'''
        #select frequent users, here you need to change the matched user files if selection criteria changed
        participants = pd.read_csv(self.path + 'participants_matched.csv')   
        text = pd.read_csv(self.path + 'status_sentiment.csv')
        text_merge = pd.merge(participants, text, on = 'userid')
        text_fea = text_merge[['userid','text']]
        text_fea['text'] = text_fea.groupby(['userid'])['text'].transform(lambda x: ''.join(str(x)))
        text_fea2 = text_fea.drop_duplicates()
        #text_fea2.to_csv(path + 'text.csv')
        return text_fea2


    def parallelize_dataframe(self, df, func):
        """
        parallelize processing pandas dataframe using multithread
        """
        df_split = np.array_split(df, self.num_partitions)
        pool = Pool(self.num_cores)
        df = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()
        return df

    def get_precocessed_text(self, file):
        """
        apply function on parallelize dataframe
        """
        file['text'] = file['text'].apply(lambda x: self.preprocess(x))
        return file

    def get_tfidf(self):
        '''get tfidf of the text file, parallel processing  '''
        text = self.prepare_text_data()
        file = self.parallelize_dataframe(text, self.get_precocessed_text)

        #here you set the maximum number of features 
        tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
        VecCounts = tfidfconverter.fit_transform(file.text).toarray()
        VecCountsDf = pd.DataFrame(VecCounts)
        VecCountsDf.columns = tfidfconverter.get_feature_names()
        VecCountsDf['userid'] = file['userid']
        #save tfidf to csv file
        VecCountsDf.to_csv(self.path + 'countVect.csv')
        return VecCounts

    
if __name__ == '__main__':
    print('initializing..')

    #path = '/home/lucia/phd_work/mypersonality_data/predicting_depression_symptoms/data/'
    #path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/'
    
    c= Count_Vect()
    #text = c.prepare_text_data()
    print('get tfidf scores and save it to csv ...')
    start = time.time()
    tfidf = c.get_tfidf()
    end = time.time()
    print('running time', end-start)

   
    






