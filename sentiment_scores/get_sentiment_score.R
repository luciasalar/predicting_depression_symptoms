#this script is to create a csv file with sentiment score on the status

#extract status from csv and convert it to txt
path = "/Users/lucia/phd_work/cognitive_distortion/data/"
setwd(path)
status <- read.csv('user_status_match_dep.csv')

status_only <- as.data.frame(status$X2)
write.table(status_only, file = "status_only.txt", sep = "")

#run sentistrength
path2 = "/Users/lucia/Desktop/sentistrength/"
setwd(path2)
command = 'java -jar SentiStrength.jar sentidata /Users/lucia/Desktop/sentistrength/SentiStrength_Data/ input /Users/lucia/phd_work/predicting_depression_symptoms/status_only.txt'
system(command,intern=TRUE)

#run sentiment_score.py to read sentistrengh txt in a proper way
library("reticulate")
# Set the path to the Python executable file
use_python("/Users/lucia/anaconda3/bin/python", required = T)
# Check the version of Python.
py_config()
path3 = "/Users/lucia/phd_work/predicting_depression_symptoms/"
setwd(path3)
py_run_file("sentiment_score.py")

#read sentistrength score
library(stringr)
path = "/Users/lucia/phd_work/predicting_depression_symptoms/"
setwd(path)
sentiment_score <- read.csv("sentiment_score.csv")
#remove rowNum is not a number
sentiment_clean <- sentiment_score[!grepl("[a-z]", sentiment_score$rowNum),]
sentiment_clean$rowNum <- lapply(sentiment_clean$rowNum, gsub, pattern='\\\"', replacement='')
sentiment_clean$rowNum <- as.numeric(sentiment_clean$rowNum)

#match row number with status file
library(data.table)
status_only <- status[c('X0','X1','X2')]
status_origin <- setDT(status_only, keep.rownames = TRUE)[]
status_origin$rn <- as.numeric(status_origin$rn)
sentiment <- sentiment_clean[c('Positive', 'Negative', 'rowNum')]
status_sentiment <- merge(status_origin, sentiment, by.x = 'rn', by.y = 'rowNum')
names(status_sentiment) <- c('rownum', 'userid','time','text', 'positive','negative')

write.csv(status_sentiment, 'status_sentiment.csv')
