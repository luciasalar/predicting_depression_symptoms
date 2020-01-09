import tensorflow as tf
import pandas as pd
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import torch 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from transformers import AdamW, BertConfig, BertTokenizer, get_linear_schedule_with_warmup
from pytorch_pretrained_bert import BertModel, BertForMaskedLM, BertForSequenceClassification
from sklearn.metrics import confusion_matrix, f1_score, precision_score,\
recall_score, confusion_matrix, classification_report, accuracy_score 
import time
import datetime
import numpy as np
import random
import pickle
import copy
from ruamel import yaml
import os
import csv
from collections import defaultdict
from sklearn.utils import shuffle
import collections
 
class Preprocess:
	def __init__(self):
		self.path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/'
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True,  sep_token='[SEP]', cls_token='[CLS]')
		self.batch_size = 32
		

	def load_experiment(self):
		#load experiment 
		experiment = yaml.safe_load(open(self.path + '../experiment/experiment_bert.yaml'))
		return experiment

	def get_kaggle_data(self):
		'''get a set of data for testing '''
		text_label = pd.read_csv(self.path + 'kaggle_data/training.1600000.processed.noemoticon.csv', encoding = "ISO-8859-1")
		#adjust labels to positive: 1, negative: -1, neutral: 0, mix: 0 
		text_label.columns = ['labels','id','date','flag','user','text']
		text_label['labels'] = text_label['labels'].replace([0, 4], [0, 1]) #recode values
		text_label = shuffle(text_label).head(10000)
		print(text_label.labels)
		#get different classes
		pd_0 = text_label.loc[text_label['labels'] == 0]
		pd_1 = text_label.loc[text_label['labels'] == 1]

        #split train 85.5%, test 5% and validation 9.5%, stratified
		np.random.seed(2019)
		test_0, train_0, valid_0 = np.split(pd_0, [int(.05*len(pd_0)), int(.885*len(pd_0))])
		test_1, train_1, valid_1 = np.split(pd_1, [int(.05*len(pd_1)), int(.885*len(pd_1))])
		
		#combine data
		test_df = test_0.append(test_1)
		

		train_df = train_0.append(train_1)
	

		valid_df = valid_0.append(valid_1)
	

		shuffle(test_df).to_csv(self.path + 'kaggle_data/kaggle_test.csv')
		shuffle(train_df).to_csv(self.path + 'kaggle_data/kaggle_train.csv')
		shuffle(valid_df).to_csv(self.path + 'kaggle_data/kaggle_validation.csv')

		return test_df, train_df, valid_df

	def get_test_data(self):
		'''get a set of data for testing '''
		text_label = pd.read_csv(self.path + 'annotate_data.csv')
		#adjust labels to positive: 1, negative: -1, neutral: 0, mix: 0 
		participants = text_label[['userid','text','negative_ny','time', 'time_diff']]
		participants=participants.rename(columns = {'negative_ny':'labels'})
		participants = participants.loc[participants['labels'] != 5,] #remove non English
		participants['labels'] = participants['labels'].replace([1, 2, 3, 4], [0, 2, 1, 1]) #recode values
		#get different classes
		pd_0 = participants.loc[participants['labels'] == 0]
		pd_1 = participants.loc[participants['labels'] == 1]
		pd_2 = participants.loc[participants['labels'] == 2]

        #split train 85.5%, test 5% and validation 9.5%, stratified
		np.random.seed(2019)
		test_0, train_0, valid_0 = np.split(pd_0, [int(.05*len(pd_0)), int(.885*len(pd_0))])
		test_1, train_1, valid_1 = np.split(pd_1, [int(.05*len(pd_1)), int(.885*len(pd_1))])
		test_2, train_2, valid_2  = np.split(pd_2, [int(.05*len(pd_2)), int(.885*len(pd_2))])
		#combine data
		test_df = test_0.append(test_1)
		test_df = test_df.append(test_2)

		train_df = train_0.append(train_1)
		train_df = train_df.append(train_2)

		valid_df = valid_0.append(valid_1)
		valid_df = valid_df.append(valid_2)

		shuffle(test_df).to_csv(self.path + 'annotate_test.csv')
		shuffle(train_df).to_csv(self.path + 'annotate_train.csv')
		shuffle(valid_df).to_csv(self.path + 'annotate_validation.csv')

		return test_df, train_df, valid_df
		#return participants

	def get_data(self, fileName):
		'''separate text and labels '''
		data = pd.read_csv(fileName)
		#data = data.head(100)
		

		#get text and label as txt file
		sentences = data.text.values
		#np.savetxt(self.path + 'text.txt', text.values, fmt='%d')
		#text.to_csv(self.path + 'text.txt', sep=' ', index=False, header=False)

		labels = data.labels.values
		#np.savetxt(self.path + 'labels.txt', labels.values, fmt='%d')
		#labels.to_csv(self.path + 'labels.txt', sep=' ', index=False, header=False)

		return sentences, labels



	def get_all_data(self, trainset, testset, validationset):
		'''get datasets and labels '''
		
		train_sentences, train_labels = self.get_data(self.path + trainset)


		# augment training data
		# paraphase = pd.read_fwf(self.path + 'train_text_paraphrase.txt', header = None)
		# paraphase = pd.read_csv(self.path + 'kaggle_data/kaggle_train.csv')
		# paraphase = paraphase.head(5000)
		# sentences_p = paraphase.text.values
		# ka_labels = paraphase.labels.values
		# append paraphrase to orginal training set
		# train_sentences = np.append(train_sentences, sentences_p, axis = 0)
		# train_labels = np.append(train_labels, ka_labels, axis = 0)

		# get test and validation set
		test_sentences, test_labels = self.get_data(self.path + testset)
		validation_sentences, validation_labels = self.get_data(self.path + validationset)

		

		return train_sentences, train_labels, test_sentences, test_labels, validation_sentences, validation_labels



	def _encode_text(self, sentences):
		
		# Tokenize all of the sentences and map the tokens to thier word IDs.
		input_ids = []

		# For every sentence...
		for sent in sentences:
		    # `encode` will:
		    #   (1) Tokenize the sentence.
		    #   (2) Prepend the `[CLS]` token to the start.
		    #   (3) Append the `[SEP]` token to the end.
		    #   (4) Map tokens to their IDs.
		    encoded_sent = self.tokenizer.encode(sent, add_special_tokens = True)
		    
		    # Add the encoded sentence to the list.
		    input_ids.append(encoded_sent)

		return input_ids

	def encode_all_text(self):
		train_sentences, train_labels, test_sentences, test_labels, validation_sentences, validation_labels = self.get_all_data('annotate_train.csv', 'annotate_test.csv', 'annotate_validation.csv')

		# train_sentences, train_labels, test_sentences, test_labels, validation_sentences, validation_labels = self.get_all_data('kaggle_data/kaggle_train.csv', 'annotate_train.csv', 'kaggle_data/kaggle_validation.csv')

		train_input_ids = self._encode_text(train_sentences)
		test_input_ids = self._encode_text(test_sentences)
		validation_input_ids = self._encode_text(validation_sentences)

		return train_input_ids, test_input_ids, validation_input_ids 


	def pad_sentences(self, MAX_LEN):
		train_input_ids, test_input_ids, validation_input_ids = self.encode_all_text()

		MAX_LEN = MAX_LEN

		print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)

		print('\nPadding token: "{:}", ID: {:}'.format(self.tokenizer.pad_token, self.tokenizer.pad_token_id))

		# Pad our input tokens with value 0.
		train_input = pad_sequences(train_input_ids, maxlen=MAX_LEN, dtype="long", 
		                          value=0, truncating="post", padding="post")

		test_input = pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype="long", 
		                          value=0, truncating="post", padding="post")

		validation_input = pad_sequences(validation_input_ids, maxlen=MAX_LEN, dtype="long", 
		                          value=0, truncating="post", padding="post")

		print('\nDone.')
		return train_input, test_input, validation_input


	def _attention_mask(self, input_ids):
		# Create attention masks
		
		attention_masks = []

		# For each sentence...
		for sent in input_ids:
		    
		    # Create the attention mask.
		    #   - If a token ID is 0, then it's padding, set the mask to 0.
		    #   - If a token ID is > 0, then it's a real token, set the mask to 1.
		    att_mask = [int(token_id > 0) for token_id in sent]
		    
		    # Store the attention mask for this sentence.
		    attention_masks.append(att_mask)
		return attention_masks

	def get_attention_masks(self, MAX_LEN):
		'''get all the attention masks '''
		train_input_ids, test_input_ids, validation_input_ids = self.pad_sentences(MAX_LEN)

		train_masks = self._attention_mask(train_input_ids)
		test_masks = self._attention_mask(test_input_ids)
		validation_masks = self._attention_mask(validation_input_ids)


		return train_masks, test_masks, validation_masks


	def data_loader(self, MAX_LEN):
		# Convert all inputs and labels into torch tensors, the required datatype 
		# for our model.
		train_sentences, train_labels, test_sentences, test_labels, validation_sentences, validation_labels = self.get_all_data('annotate_train.csv', 'annotate_test.csv', 'annotate_validation.csv')
		#

		# train_sentences, train_labels, test_sentences, test_labels, validation_sentences, validation_labels = self.get_all_data('kaggle_data/kaggle_train.csv', 'annotate_train.csv', 'kaggle_data/kaggle_validation.csv')
		train_inputs, test_inputs, validation_inputs = self.pad_sentences(MAX_LEN)
		train_masks, test_masks, validation_masks = self.get_attention_masks(MAX_LEN)


		train_inputs = torch.LongTensor(train_inputs)
		validation_inputs = torch.LongTensor(validation_inputs)
		test_inputs = torch.LongTensor(test_inputs)

		train_labels = torch.LongTensor(train_labels)
		validation_labels = torch.LongTensor(validation_labels)
		test_labels = torch.LongTensor(test_labels)

		train_masks = torch.LongTensor(train_masks)
		validation_masks = torch.LongTensor(validation_masks)
		test_masks = torch.LongTensor(test_masks)

		# The DataLoader needs to know our batch size for training, so we specify it 
		# here.
		# For fine-tuning BERT on a specific task, the authors recommend a batch size of
		# 16 or 32.

		
		# Create the DataLoader for our training set.
		print('train labels length:', len(train_labels))
		train_data = TensorDataset(train_inputs, train_masks, train_labels)
		train_sampler = RandomSampler(train_data)
		train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

		# Create the DataLoader for our validation set.
		validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
		validation_sampler = SequentialSampler(validation_data)
		validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=self.batch_size)

		# Create the DataLoader for our validation set.
	
		test_data = TensorDataset(test_inputs, test_masks, test_labels)
		test_sampler = SequentialSampler(test_data)
		test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size)



		# The DataLoader needs to know our batch size for training, so we specify it 
		# here.
		# For fine-tuning BERT on a specific task, the authors recommend a batch size of
		# 16 or 32.
		# Create the DataLoader for our training set.
		return train_dataloader, validation_dataloader, test_dataloader

class Training:
	def __init__(self, path, train_dataloader, validation_dataloader, test_dataloader, device):

		self.path = path
		self.train_dataloader = train_dataloader
		self.validation_dataloader = validation_dataloader
		self.test_dataloader = test_dataloader
		self.device = device

	def save_model(self, model):
		saved_trainer = copy.deepcopy(model)
		with open(self.path + "embeddings/kaggle_model.pkl", "wb") as output_file:
		    pickle.dump(saved_trainer, output_file)

	def load_model(self, modelName):
		with open(self.path + modelName, "rb") as input_file:
		    model = pickle.load(input_file)
		return model

	def change_neutral_criterion1(self, preds, threshold):
		'''this criterion is for ordinal regression of annotation data'''
		prev_row = None
		
		count = 0
		for row in preds:
			if row[1] > threshold:
				row[1] = 1
				# print('T')
			if count == 0:
				prev_row = np.array(row)


			else:
				prev_row = np.vstack((prev_row, np.array(row)))
	
			count = count + 1
		return prev_row

	def change_neutral_criterion2(self, preds, threshold):
		'''this criterion is for trained sentiment140 model used on annotated data'''
		prev_row = None
		
		count = 0
		for row in preds:
			if abs(row[0] - row[1]) < 0.1:
				np.append(row, 1)
			else:
				np.append(row, 0)
				# print('T')
			if count == 0:
				prev_row = np.array(row)


			else:
				prev_row = np.vstack((prev_row, np.array(row)))
	
			count = count + 1
		return prev_row


		# Function to calculate the accuracy of our predictions vs labels
	def flat_accuracy(self, preds, labels):
		'''compute accuracy '''

		pred_flat = np.argmax(preds, axis=1).flatten()
		
		labels_flat = labels.flatten()

		print(pred_flat, labels_flat)
		return np.sum(pred_flat == labels_flat) / len(labels_flat)

	def flat_accuracy_change(self, preds, labels, threshold):
		'''compute accuracy of changed criterion '''

		# change neutral class value to 1 if it's value > 0.21
		new_pred = self.change_neutral_criterion2(preds, threshold)
		# print(new_pred.shape)
		# print(preds.shape)

		pred_flat = np.argmax(new_pred, axis=1).flatten()
		
		labels_flat = labels.flatten()

		print(pred_flat, labels_flat)
		return np.sum(pred_flat == labels_flat) / len(labels_flat)

	def report(self, preds, labels):

	
		report = classification_report(labels, np.argmax(preds, axis = 1).flatten(), output_dict=True)

		y_pred_series = pd.DataFrame(np.argmax(new_pred, axis = 1).flatten())
		result = pd.concat([pd.Series(labels).reset_index(drop=True), y_pred_series], axis = 1)
		result.columns = ['y_true', 'y_pred']
		#result.to_csv(path + 'results/best_result2.csv' )
		return report

	def report_change(self, preds, labels, threshold):
		new_pred = self.change_neutral_criterion2(preds, threshold)
	
		report = classification_report(labels, np.argmax(new_pred, axis = 1).flatten(), output_dict=True)

		y_pred_series = pd.DataFrame(np.argmax(new_pred, axis = 1).flatten())
		result = pd.concat([pd.Series(labels).reset_index(drop=True), y_pred_series], axis = 1)
		result.columns = ['y_true', 'y_pred']
		#result.to_csv(path + 'results/best_result2.csv' )
		return report


	def format_time(self, elapsed):
	    '''
	    Takes a time in seconds and returns a string hh:mm:ss
	    '''
	    # Round to the nearest second.
	    elapsed_rounded = int(round((elapsed)))
	    
	    # Format as hh:mm:ss
	    return str(datetime.timedelta(seconds=elapsed_rounded))

	def train(self, learning_rate, clipping):

		seed_val = 42

		random.seed(seed_val)
		np.random.seed(seed_val)
		torch.manual_seed(seed_val)
		torch.cuda.manual_seed_all(seed_val)

		model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

		params = list(model.named_parameters())

		optimizer = AdamW(model.parameters(),
		                  lr = learning_rate, # args.learning_rate - default is 5e-5, our notebook had 2e-5
		                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
		                )


		# Number of training epochs (authors recommend between 2 and 4)
		epochs = 4

		# Total number of training steps is number of batches * number of epochs.
		total_steps = len(self.train_dataloader) * epochs

		# Create the learning rate scheduler.
		scheduler = get_linear_schedule_with_warmup(optimizer, 
		                                            num_warmup_steps = 0, # Default value in run_glue.py
		                                            num_training_steps = total_steps)

		# Set the seed value all over the place to make this reproducible.


		# Store the average loss after each epoch so we can plot them.
		loss_values = []

		model.zero_grad()

		#For each epoch...
		for epoch_i in range(0, epochs):
		    
		    # ========================================
		    #               Training
		    # ========================================
		    
		    # Perform one full pass over the training set.

		    print("")
		    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
		    print('Training...')

		    # Measure how long the training epoch takes.
		    t0 = time.time()

		    # Reset the total loss for this epoch.
		    total_loss = 0

		    # Set our model to training mode (as opposed to evaluation mode)
		    model.train()
		        
		    # This training code is based on the `run_glue.py` script here:
		    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
		    print('start training')
		    # For each batch of training data...
		    for step, batch in enumerate(self.train_dataloader):

		        # Progress update every 40 batches.
		        if step % 40 == 0 and not step == 0:
		            # Calculate elapsed time in minutes.
		            elapsed = self.format_time(time.time() - t0)
		            
		            # Report progress.
		            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_dataloader), elapsed))

		        # Put the model into training mode.    
		        model.train()
		        print('Put the model into training mode. ')
		        # Unpack this training batch from our dataloader. 
		        #
		        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
		        # `to` method.
		        #
		        # `batch` contains three pytorch tensors:
		        #   [0]: input ids 
		        #   [1]: attention masks
		        #   [2]: labels 
		       	print('copy each tensor to the GPU')
		        b_input_ids = batch[0].to(self.device)
		        b_input_mask = batch[1].to(self.device)
		        b_labels = batch[2].to(self.device)
		                
		        # Forward pass (evaluate the model on this training batch)
		        # `model` is of type: pytorch_pretrained_bert.modeling.BertForSequenceClassification
		        outputs = model(b_input_ids, 
		                    token_type_ids=None, 
		                    attention_mask=b_input_mask, 
		                    labels=b_labels)

		        # dropout= torch.nn.Dropout(0.5)
		        # outputs = dropout(outputs)

		        print('forward pass')
		       	# print(b_input_ids)
		       	# print(b_input_mask)
		        # print(outputs)
		        #loss = outputs[0]
		        loss = outputs
		        # Accumulate the loss. `loss` is a Tensor containing a single value; 
		        # the `.item()` function just returns the Python value from the tensor.
		        total_loss += loss.item()

		        # Perform a backward pass to calculate the gradients.
		        loss.backward()

		        # Clip the norm of the gradients to 1.0.
		        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)

		        # Update parameters and take a step using the computed gradient
		        optimizer.step()

		        # Update the learning rate.
		        scheduler.step()

		        # Clear out the gradients (by default they accumulate)
		        model.zero_grad()

		    # Calculate the average loss over the training data.
		    avg_train_loss = total_loss / len(self.train_dataloader)            
		    
		    loss_values.append(avg_train_loss)


		    #self.save_model(model)
		    torch.save(model, self.path + "embeddings/kaggle_pretrained.pt" )

		    print("")
		    print("  Average training loss: {0:.2f}".format(avg_train_loss))
		    print("  Training epcoh took: {:}".format(self.format_time(time.time() - t0)))

		    return model
		        
		    # ========================================
		    #               Validation
		    # ========================================
		    # After the completion of each training epoch, measure our performance on
		    # our validation set.
	def test_on_annotation(self, lr, clipping, threshold):
	    print("")
	    print("Running Validation...")

	    t0 = time.time()
	    model = self.load_model("embeddings/kaggle_model.pkl")
	   
	    model = self.train(learning_rate = lr, clipping = clipping)
	    # Put model in evaluation mode to evaluate loss on the validation set
	    model.eval()

	    # Tracking variables 
	    eval_loss, eval_accuracy = 0, 0
	    nb_eval_steps, nb_eval_examples = 0, 0
	    class_0_pre, class_0_re, class_0_f1, class_1_pre, class_1_re, class_1_f1, class_2_pre, class_2_re, class_2_f1= 0, 0, 0, 0, 0, 0, 0, 0, 0 

	    # Evaluate data for one epoch

	    for batch in self.test_dataloader:
	        
	        # Add batch to GPU
	        batch = tuple(t.to(self.device) for t in batch)
	        
	        # Unpack the inputs from our dataloader
	        b_input_ids, b_input_mask, b_labels = batch
	        
	        # Telling the model not to compute or store gradients, saving memory and speeding up validation
	        with torch.no_grad():        
	            # Forward pass, calculate logit predictions
	            # token_type_ids is for the segment ids, but we only have a single sentence here.
	            # See https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L258 
	            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
	        
	        logits = outputs


	        # Move logits and labels to CPU
	        logits = logits.detach().cpu().numpy()
	        label_ids = b_labels.to('cpu').numpy()
	        
	        # Calculate the accuracy for this batch of test sentences.
	        tmp_eval_accuracy = self.flat_accuracy_change(logits, label_ids, threshold)
	        report = self.report_change(logits, label_ids, threshold)
	        # print(logits)
	        # print(label_ids)
	        # print(report)
	        # print(tmp_eval_accuracy)
	        
	        # Accumulate the total accuracy.
	        eval_accuracy += tmp_eval_accuracy
	        # print(tmp_eval_accuracy)
	        class_0_pre += report['0']['precision']
	        class_0_re += report['0']['recall']
	        class_0_f1 += report['0']['f1-score']

	        class_1_pre += report['1']['precision']
	        class_1_re += report['1']['recall']
	        class_1_f1 += report['1']['f1-score']

	        class_2_pre += report['2']['precision']
	        class_2_re += report['2']['recall']
	        class_2_f1 += report['2']['f1-score']

	        # Track the number of batches
	        nb_eval_steps += 1

	    # Report the final accuracy for this validation run.
	    mydict = lambda: defaultdict(mydict)
	    report_dict = mydict()

	    accuracy = eval_accuracy/nb_eval_steps
	    class_0_precision = class_0_pre/nb_eval_steps
	    class_0_recall = class_0_re/nb_eval_steps
	    class_0_f1score = class_0_f1/nb_eval_steps

	    class_1_precision = class_1_pre/nb_eval_steps
	    class_1_recall = class_1_re/nb_eval_steps
	    class_1_f1score = class_1_f1/nb_eval_steps

	    class_2_precision = class_2_pre/nb_eval_steps
	    class_2_recall = class_2_re/nb_eval_steps
	    class_2_f1score = class_2_f1/nb_eval_steps

	    report_dict['0']['precision'] = class_0_precision
	    report_dict['0']['recall'] = class_0_recall
	    report_dict['0']['f1-score'] = class_0_f1score

	    report_dict['1']['precision'] = class_1_precision
	    report_dict['1']['recall'] = class_1_recall
	    report_dict['1']['f1-score'] = class_1_f1score

	    report_dict['2']['precision'] = class_2_precision
	    report_dict['2']['recall'] = class_2_recall
	    report_dict['2']['f1-score'] = class_2_f1score

	    print("  Accuracy: {0:.2f}".format(accuracy))
	    print("  report: {}".format(pd.DataFrame(report_dict)))
	    print("  Validation took: {:}".format(self.format_time(time.time() - t0)))

	    print("")
	    print("Training complete!")
	    return accuracy, report_dict

		

	def validation(self, lr, clipping):
	    print("")
	    print("Running Validation...")
	    t0 = time.time()
	    #model = self.load_model()
	   
	    model = self.train(learning_rate = lr, clipping = clipping)
	    # Put model in evaluation mode to evaluate loss on the validation set
	    model.eval()

	    # Tracking variables 
	    eval_loss, eval_accuracy = 0, 0
	    nb_eval_steps, nb_eval_examples = 0, 0
	    class_0_pre, class_0_re, class_0_f1, class_1_pre, class_1_re, class_1_f1, class_2_pre, class_2_re, class_2_f1= 0, 0, 0, 0, 0, 0, 0, 0, 0 

	    # Evaluate data for one epoch

	    for batch in self.validation_dataloader:
	        
	        # Add batch to GPU
	        batch = tuple(t.to(self.device) for t in batch)
	        
	        # Unpack the inputs from our dataloader
	        b_input_ids, b_input_mask, b_labels = batch
	        
	        # Telling the model not to compute or store gradients, saving memory and speeding up validation
	        with torch.no_grad():        
	            # Forward pass, calculate logit predictions
	            # token_type_ids is for the segment ids, but we only have a single sentence here.
	            # See https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L258 
	            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
	        
	        logits = outputs


	        # Move logits and labels to CPU
	        logits = logits.detach().cpu().numpy()
	        label_ids = b_labels.to('cpu').numpy()
	        
	        # Calculate the accuracy for this batch of test sentences.
	        tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)
	        report = self.report(logits, label_ids)
	        # print(logits)
	        # print(label_ids)
	        # print(report)
	        # print(tmp_eval_accuracy)
	        
	        # Accumulate the total accuracy.
	        eval_accuracy += tmp_eval_accuracy
	        # print(tmp_eval_accuracy)
	        class_0_pre += report['0']['precision']
	        class_0_re += report['0']['recall']
	        class_0_f1 += report['0']['f1-score']

	        class_1_pre += report['1']['precision']
	        class_1_re += report['1']['recall']
	        class_1_f1 += report['1']['f1-score']

	        class_2_pre += report['2']['precision']
	        class_2_re += report['2']['recall']
	        class_2_f1 += report['2']['f1-score']

	        # Track the number of batches
	        nb_eval_steps += 1

	    # Report the final accuracy for this validation run.
	    mydict = lambda: defaultdict(mydict)
	    report_dict = mydict()

	    accuracy = eval_accuracy/nb_eval_steps
	    class_0_precision = class_0_pre/nb_eval_steps
	    class_0_recall = class_0_re/nb_eval_steps
	    class_0_f1score = class_0_f1/nb_eval_steps

	    class_1_precision = class_1_pre/nb_eval_steps
	    class_1_recall = class_1_re/nb_eval_steps
	    class_1_f1score = class_1_f1/nb_eval_steps

	    class_2_precision = class_2_pre/nb_eval_steps
	    class_2_recall = class_2_re/nb_eval_steps
	    class_2_f1score = class_2_f1/nb_eval_steps

	    report_dict['0']['precision'] = class_0_precision
	    report_dict['0']['recall'] = class_0_recall
	    report_dict['0']['f1-score'] = class_0_f1score

	    report_dict['1']['precision'] = class_1_precision
	    report_dict['1']['recall'] = class_1_recall
	    report_dict['1']['f1-score'] = class_1_f1score

	    report_dict['2']['precision'] = class_2_precision
	    report_dict['2']['recall'] = class_2_recall
	    report_dict['2']['f1-score'] = class_2_f1score

	    print("  Accuracy: {0:.2f}".format(accuracy))
	    print("  report: {}".format(pd.DataFrame(report_dict)))
	    print("  Validation took: {:}".format(self.format_time(time.time() - t0)))

	    print("")
	    print("Training complete!")
	    return accuracy, report_dict

def get_device():
	#If there's a GPU available...
	if torch.cuda.is_available():    

	    # Tell PyTorch to use the GPU.    
	    device = torch.device("cuda")

	    print('There are %d GPU(s) available.' % torch.cuda.device_count())

	    print('We will use the GPU:', torch.cuda.get_device_name(0))

	# If not...
	else:
	    print('No GPU available, using the CPU instead.')
	    device = torch.device("cpu")
	return device



if __name__ == "__main__":
	pre = Preprocess()
	path = pre.path
	MAX_LEN = 32
	# a, b, c = pre.get_test_data()
	# train_sentences, train_labels, test_sentences, test_labels, validation_sentences, validation_labels = pre.get_all_data('annotate_train.csv', 'annotate_test.csv', 'annotate_validation.csv')

	train_dataloader, validation_dataloader, test_dataloader = pre.data_loader(MAX_LEN)

	device = get_device()

	t= Training(path = path, train_dataloader = train_dataloader, validation_dataloader = validation_dataloader, test_dataloader = test_dataloader, device = device)

	experiment = pre.load_experiment()

	def loop_the_grid(path, experiment, MAX_LEN):
		

		file_exists = os.path.isfile(path + 'results/result_bert.csv')
		f = open(path + 'results/result_bert.csv' , 'a')
		writer_top = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
		if not file_exists:
			writer_top.writerow(['accuracy'] +['report'] +['time'] + ['learning_rate'] + ['gradient_clipping'] + ['MAX_LEN'] + ['threshold'])
			f.close()
			

		for lr in experiment['experiment']['learning_rate']:
			for clip in experiment['experiment']['clipping']:
				for threshold in experiment['experiment']['threshold']:

					f = open(path + 'results/result_bert.csv' , 'a')
					writer_top = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
					#model = t.train(learning_rate = lr)
					tmp_eval_accuracy, report = t.validation(lr, clip)
					#tmp_eval_accuracy, report = t.test_on_annotation(lr, clip, threshold)

					result_row = [[tmp_eval_accuracy, pd.DataFrame(report), str(datetime.datetime.now()), lr, clip, MAX_LEN, threshold]]

					writer_top.writerows(result_row)

					f.close()


	loop_the_grid(pre.path, experiment, 64)

# tmp_eval_accuracy, report = t.validation(lr, clip, test_dataloader)

# for lr in experiment['experiment']['learning_rate']:
# 	for clip in experiment['experiment']['clipping']:
# 		print(lr, clip)


#model = t.train(0.0001)
#t.validation(0.0001)

# train_txt = pd.read_csv(pre.path + 'annotate_train.csv')

# train_txt['text'].to_csv(pre.path + 'train_text.txt', sep=' ', index=False, header=False)
