import pandas as pd 


def sentiment_groups(series):
	'''recode sentiment to categorical '''
    if series > 0:
        return 2
    elif  series < 0:
        return 1
    elif  series == 0:
        return 4


#read sentiment file
path = '/home/lucia/phd_work/mypersonality_data/predicting_depression_symptoms/'
sentiment_status = pd.read_csv('status_sentiment.csv')

#mapping sentiment score (positive: 2, negative:1, neutral:4)
'''if sum positive, negaitve = 0 -> neutral   > 0 positive, < 0 negative '''
sentiment_status['sentiment_sum'] = sentiment_status['positive'] + sentiment_status['negative']
sentiment_status['class'] = sentiment_status['sentiment_sum'].apply(lambda x: sentiment_groups(x))