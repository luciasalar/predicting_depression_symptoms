# data and result

/data
contains data for this project, data is stored in hawksworth:share/predicting_depression_symptoms/data
I move the data to myCloud but all the paths in the scripts point to hawksworth

/data/results
contains results for the models

# data selection

script for data selection process

# sentiment score

scripts to retrieve sentiment score

construct_mood_feature.py
This script return mood feature matrix (mood (both categorical and continuous) and mood momentum) for the prediction model

construct_mood_feature.py
return mood transition feature matrix (mood transitions and mood transition momentum) for the prediction model

model_training.py, run_model.sh
Prediction model on depression symptoms (binary)
run_model.sh is the bash script for running the model in longjob 

CountVec.py
some data preprocessing functions

GP_process.py
GP regression fit on each user

GP_stats.py
Statistics analysis of GP result (some t-tests)

HMMmodeling.py
HMM models

topic_model.py
return topic score matrix as feature

Other script:
fine_tune_bert.py
train sentiment score with fine tune bert model (this part is not used in the paper because the model is slightly overfitting)


QuoteDetector.py
check if a post is a quotation or lyric

QuoteDetector.py
check if a post is a quotation or lyric

resample_data.py
resample the data using various conditions

plotting_results.py, plot_results.Rmd
plot results from recall and precision tables





