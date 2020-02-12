"""This script examine the relationship between affective quote and depression"""
from quotation_feature import *
import pandas as pd 
import datetime
import collections 
import numpy as np
import scipy.stats
import statsmodels.api as sm
from topic_model import *
from CountVect import *

class stats:
    def __init__(self, days):

       self.qd = QuotationFeaDynamic()
       self.days = days
       self.path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/'

    def get_data_by_day(self):
        """get posts from the recent X days"""

        sorted_quote = self.qd.get_relative_day(self.days)
        return sorted_quote

    def count_label(self):
        """count number of posts in the past x days"""

        sorted_quote = self.get_data_by_day()
        label_c = sorted_quote.groupby('label').count()
        return label_c

    def quote_cor(self):
        """count average sentiment peruser, correlation between affective quote and depression"""
        sorted_quote = self.get_data_by_day()

        pos1 = sorted_quote.loc[sorted_quote['sentiment_sum'] > 0]
        neg1 = sorted_quote.loc[sorted_quote['sentiment_sum'] < 0]
        neu1 = sorted_quote.loc[sorted_quote['sentiment_sum'] == 0]

        pos = pos1.groupby(['userid']).size().to_frame(name='pos').reset_index()
        neg = neg1.groupby(['userid']).size().to_frame(name='neg').reset_index()
        neu = neu1.groupby(['userid']).size().to_frame(name='neu').reset_index()
        
        # get non-orginal content
        quote = sorted_quote.loc[sorted_quote['label'] == 1]
        # sorted_quote.to_csv(self.path + 'quotation_f.csv')
        # depression score
        cesd = quote.drop_duplicates(subset='userid', keep="last")
        cesd = cesd[['userid', 'cesd_sum']]
        # count number of non-orginal content
        all_count = quote.groupby(['userid']).size().to_frame(name='all_count').reset_index()
        all_count = cesd.merge(all_count,on='userid')
        # count number of positive non-original content
        positive = quote.loc[quote['sentiment_sum'] > 0]
        per_pos = positive.groupby(['userid']).size().to_frame(name='pos_counts').reset_index()

        negative = quote.loc[quote['sentiment_sum'] < 0]
        per_nega = negative.groupby(['userid']).size().to_frame(name='nega_counts').reset_index()

        neutral = quote.loc[quote['sentiment_sum'] == 0]
        per_neu = neutral.groupby(['userid']).size().to_frame(name='neu_counts').reset_index()

        lyrics = quote.loc[quote['tag'] == 1]
        lyrics_c = lyrics.groupby(['userid']).size().to_frame(name='lyrics').reset_index()

        pos_lyrics1 = lyrics.loc[lyrics['sentiment_sum'] > 0]
        neg_lyrics1 = lyrics.loc[lyrics['sentiment_sum'] < 0]
        neu_lyrics1 = lyrics.loc[lyrics['sentiment_sum'] == 0]
        
        pos_lyrics = pos_lyrics1.groupby(['userid']).size().to_frame(name='pos_lyrics').reset_index()
        neg_lyrics = neg_lyrics1.groupby(['userid']).size().to_frame(name='neg_lyrics').reset_index()
        neu_lyrics = neu_lyrics1.groupby(['userid']).size().to_frame(name='neu_lyrics').reset_index()
        
        quote2 = quote.loc[quote['tag'] == 2]
        pos_quote = quote2.loc[quote2['sentiment_sum'] > 0]
        neg_quote = quote2.loc[quote2['sentiment_sum'] < 0]
        neu_quote = quote2.loc[quote2['sentiment_sum'] == 0]

        pos_quote = pos_quote.groupby(['userid']).size().to_frame(name='pos_quote').reset_index()
        neg_quote = neg_quote.groupby(['userid']).size().to_frame(name='neg_quote').reset_index()
        neu_quote = neu_quote.groupby(['userid']).size().to_frame(name='neu_quote').reset_index()

        quote3 = quote2.groupby(['userid']).size().to_frame(name='quote').reset_index()

        print('Among {} non-orginal content, positive content {}, negative content {}, neutral content {},  number of quotes: {},  numeber of lyrics: {}, positive lyrics: {}, negative lyrics {}, neutral lyrics {} '.format(quote.shape[0], positive.shape[0], negative.shape[0], neutral.shape[0], quote2.shape[0], lyrics.shape[0], pos_lyrics1.shape[0], neg_lyrics1.shape[0], neu_lyrics1.shape[0]))

        return per_pos, per_nega, per_neu, all_count, lyrics_c, quote3, pos_lyrics, neg_lyrics, neu_lyrics, pos_quote, neg_quote, neu_quote, pos, neg, neu
        
    def get_count_quote(self):
        '''Here we see the count of valenced post in each user'''
        p, nega, neutral, all_count, ly, quo, pos_ly, neg_ly, neu_ly, pos_quo, neg_quo, neu_quo, pos, neg, neu = self.quote_cor()
         # merge all the counts as feature '''
        quotation_fea = p.merge(nega, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(neutral, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(all_count, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(ly, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(quo, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(pos_ly, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(neg_ly, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(neu_ly, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(pos_quo, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(neg_quo, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(neu_quo, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(pos, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(neg, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(neu, on='userid', how='outer')
        quotation_fea = quotation_fea[['userid', 'pos_counts', 'nega_counts', 'neu_counts','all_count','cesd_sum', 'lyrics','quote', 'pos_lyrics', 'neg_lyrics', 'neu_lyrics', 'pos_quote', 'neg_quote', 'neu_quote', 'pos', 'neg', 'neu']]
        quotation_fea = quotation_fea.fillna(0)
        # quotation_fea['pos_counts'] = quotation_fea['pos_counts'] / quotation_fea['all_count']
        # quotation_fea['nega_counts'] = quotation_fea['nega_counts'] / quotation_fea['all_count']
        # quotation_fea['neu_counts'] = quotation_fea['neu_counts'] / quotation_fea['all_count']
        
        return quotation_fea

    def regression(self, pre_var):
        s = stats(self.days) 
        data = s.get_count_quote()
   
        X = data[pre_var]
        y = data["cesd_sum"]
        model = sm.OLS(y, X).fit()

        predictions = model.predict(X)
        print(model.summary())
    
    
    def get_lda(self):
        """get lda topics"""

        s = stats(180)
        post = s.get_data_by_day()
        nonOrg = post.loc[post['label'] == 1]
        lyrics = nonOrg.loc[nonOrg['tag'] == 2]

        topic = LDATopicModel()
        c = Count_Vect()

        text = lyrics[['text','userid']]
        text['text'] = text['text'].apply(lambda x: c.remove_noise(str(x)))
        text= c.get_precocessed_text(text)
        topics, model = topic.get_lda_score(text, 10)
        return topics, model
   
       



# sorted_quote = s.get_data_by_day(365)
# quote = sorted_quote.loc[sorted_quote['label'] == 1]
# s.count_label(365)

# LDA topic 




#the correlation between number of posting frequency and cesd are correlated in differe time scale
if __name__ == "__main__":
	s = stats(365) 
	fea = s.get_count_quote()
	scipy.stats.pearsonr(fea['nega_counts'], fea['cesd_sum']) 
	# #(0.07944315153005584, 0.06507674491955265)
	scipy.stats.pearsonr(fea['all_count'], fea['cesd_sum'])  
	#(0.07276927041827762, 0.09115683798750919)
	scipy.stats.pearsonr(fea['pos_counts'], fea['cesd_sum']) 
	#(0.07161373241541387, 0.09642630203310558)
	scipy.stats.pearsonr(fea['lyrics'], fea['cesd_sum']) 
	scipy.stats.pearsonr(fea['quote'], fea['cesd_sum']) 

	s = stats(180) 
	fea = s.get_count_quote()
	scipy.stats.pearsonr(fea['nega_counts'], fea['cesd_sum']) 
	# (0.10523409964273388, 0.025264913426267407)
	scipy.stats.pearsonr(fea['all_count'], fea['cesd_sum'])  
	#(0.12669309837805479, 0.0069977946611849765)
	scipy.stats.pearsonr(fea['pos_counts'], fea['cesd_sum']) 
	#(0.125986557693915, 0.007322995127557867)
	scipy.stats.pearsonr(fea['lyrics'], fea['cesd_sum']) 
	scipy.stats.pearsonr(fea['quote'], fea['cesd_sum']) 


	s = stats(90) 
	fea = s.get_count_quote()
	scipy.stats.pearsonr(fea['nega_counts'], fea['cesd_sum']) 
	#(0.10028023263021851, 0.06636545630452054) 
	scipy.stats.pearsonr(fea['all_count'], fea['cesd_sum'])  
	#(0.13483902795035824, 0.013370828434891168)
	scipy.stats.pearsonr(fea['pos_counts'], fea['cesd_sum']) 
	#(0.12729545618399066, 0.019586720272746216)
	scipy.stats.pearsonr(fea['lyrics'], fea['cesd_sum']) 
	scipy.stats.pearsonr(fea['quote'], fea['cesd_sum']) 

	#let's do a regression 

	s = stats(180) 
	predictive_var = ["all_count", 'pos', "neu_lyrics"]
	#predictive_var = ['pos_lyrics', 'neg_lyrics', "neu_lyrics"]
	#predictive_var = ['pos_quote', 'neg_quote', 'neu_quote']
	#predictive_var = ['all_count']
	#predictive_var = ['pos', 'neg', 'neu']
	s.regression(predictive_var)

	# topic model 

	#in the past one year, there are 94441 posts, 5530 are quotes, 88911 are non quotes
	s = stats(90)
	topics, model = s.get_lda()