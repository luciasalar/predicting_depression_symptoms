import pandas as pd

#read txt file and save it to csv
path = "/Users/lucia/phd_work/predicting_depression_symptoms/"
data = pd.read_csv(path + 'status_only0_out.txt', sep='delimiter', header=None)
status = pd.DataFrame(data[0].str.split('\t',2).tolist(),columns = ['flips','row','three'])
new_header = status.iloc[0]
status.columns = new_header
status = status.iloc[1:]
#seperate row name
status2 = pd.DataFrame(status.Text.str.split('""',1).tolist(),columns = ['rowNum','text'])
#concatenate 
status['Text'] = status2['text']
status['rowNum'] = status2['rowNum']



status.to_csv(path+'sentiment_score.csv')