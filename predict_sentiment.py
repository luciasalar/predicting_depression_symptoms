from string import punctuation
from collections import Counter
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from torch import optim
import pickle
import copy 

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification

class Preprocess:
	def __init__(self):
		self.path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/'

	def save_txt(self):
		'''separate text and labels '''
		text_label = pd.read_csv(self.path + 'annotate_data.csv')

		#adjust labels to positive: 1, negative: -1, neutral: 0, mix: 0 
		participants = text_label[['userid','text','negative_ny','time', 'time_diff']]
		participants=participants.rename(columns = {'negative_ny':'labels'})
		participants = participants.loc[participants['labels'] != 5,] #remove non English
		participants = participants.replace([1, 2, 0, 4], [-1, 1, 0, 0]) #recode values

		#get text and label as txt file
		text = participants[['text']]
		#np.savetxt(self.path + 'text.txt', text.values, fmt='%d')
		text.to_csv(self.path + 'text.txt', sep=' ', index=False, header=False)

		labels = participants[['labels']]
		#np.savetxt(self.path + 'labels.txt', labels.values, fmt='%d')
		labels.to_csv(self.path + 'labels.txt', sep=' ', index=False, header=False)

		return text, labels

	def preprocess_text(self):
		'''preprocess text '''
		with open(self.path + 'text.txt', 'r') as f:
			text = f.read()

		text = text.lower()
		text = ''.join([c for c in text if c not in punctuation])

		## separate out individual post and store them as individual list elements. 
		text_split = text.split('\n')

		return text_split, text

	def mapping_dict(self):
		text_split, text = self.preprocess_text()

		#all_text2 = ''.join(text)
		# create a list of words
		words = text.split()
		# Count all the words using Counter Method
		count_words = Counter(words)

		total_words = len(words)
		sorted_words = count_words.most_common(total_words)

		vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}

		return vocab_to_int

	def encode_text(self):
		'''create an encoding of reviews'''
		vocab_to_int = self.mapping_dict()
		text_split, text = self.preprocess_text()

		text_ints = []
		for t in text_split:
			r = [vocab_to_int[w] for w in t.split()]
			text_ints.append(r)
			#print('Unique words: ', len((vocab_to_int))) 
		return text_ints 

	def encode_labels(self, labels):
		encoded_labels = np.array(labels['labels'].values)

		return encoded_labels


	def map_token_bert(self):
		# Tokenize all of the sentences and map the tokens to thier word IDs.
		input_ids = []
		sentences, text = self.preprocess_text()
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

		# For every sentence...
		for sent in sentences:
		    # `encode` will:
		    #   (1) Tokenize the sentence.
		    #   (2) Prepend the `[CLS]` token to the start.
		    #   (3) Append the `[SEP]` token to the end.
		    #   (4) Map tokens to their IDs.
		    encoded_sent = tokenizer.convert_tokens_to_ids(
		                        tokenizer.tokenize(sent)  # Sentence to encode.
		                       # add_special_tokens = True, # Add '[CLS]' and '[SEP]'

		                        # This function also supports truncation and conversion
		                        # to pytorch tensors, but we need to do padding, so we
		                        # can't use these features :( .
		                        #max_length = 128,          # Truncate all sentences.
		                        #return_tensors = 'pt',     # Return pytorch tensors.
		                   )
		    
		    # Add the encoded sentence to the list.
		    input_ids.append(encoded_sent)

		return input_ids

	def filter_posts(self, labels):
		#text_ints = self.encode_text()
		text_ints = self.map_token_bert()
		encoded_labels = self.encode_labels(labels)

		text_len = [len(x) for x in text_ints]
		text_ints2 = [text_ints[i] for i, l in enumerate(text_len) if l>0 & l < 100]

		
		labels2 = [encoded_labels[i] for i, l in enumerate(text_len) if l> 0 & l < 100]
		labels2 = np.asarray(labels2)

		return text_ints2, labels2




class PreTraining:
	def __init__(self, text_ints, labels, seq_length, batch_size):
		self.seq_length = seq_length
		self.text_ints = text_ints
		self.encoded_labels = labels
		self.batch_size = batch_size

	def pad_features(self):
	    ''' Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
	    '''
	    features = np.zeros((len(self.text_ints), self.seq_length), dtype = int)
	    
	    for i, post in enumerate(self.text_ints):
	        post_len = len(post)
	        
	        if post_len <= self.seq_length:
	            zeroes = list(np.zeros(self.seq_length-post_len))
	            new = zeroes+post
	        elif post_len > self.seq_length:
	            new = post[0:self.seq_length]
	        
	        features[i,:] = np.array(new)
	    
	    return features

	def split_train_test(self):

		features = self.pad_features()
		#model training
		split_frac = 0.8

		## split data into training, validation, and test data (features and labels, x and y)
		split_idx = int(len(features)*0.8)
		train_x, remaining_x = features[:split_idx], features[split_idx:]
		train_y, remaining_y = self.encoded_labels[:split_idx], self.encoded_labels[split_idx:]

		test_idx = int(len(remaining_x)*0.5)
		val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
		val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

		## print out the shapes of your resultant feature data
		print("\t\t\tFeatures Shapes:")
		print("Train set: \t\t{}".format(train_x.shape),
		      "\nValidation set: \t{}".format(val_x.shape),
		      "\nTest set: \t\t{}".format(test_x.shape))

		return train_x, train_y, val_x, val_y, test_x, test_y

	def create_tensors(self):
		train_x, train_y, val_x, val_y, test_x, test_y = self.split_train_test()

		# dataloaders
		train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
		valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
		test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))


		# make sure to SHUFFLE your data
		train_loader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
		valid_loader = DataLoader(valid_data, shuffle=True, batch_size=self.batch_size)
		test_loader = DataLoader(test_data, shuffle=True, batch_size=self.batch_size)

		return train_data, valid_data, test_data, train_loader, valid_loader, test_loader

class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)
        print(f"batch_size={batch_size}")
        # embeddings and lstm_out		
        embeds = self.embedding(x)
        print(embeds.shape)
        lstm_out, hidden = self.lstm(embeds, hidden)

        
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        train_on_gpu=torch.cuda.is_available()
        if(train_on_gpu):
          hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                   weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
          
        else:
          hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                   weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
          #print(f'n_layers = {self.n_layers}, batch_size ={batch_size}, hidden_dim = {self.hidden_dim}')
        
        return hidden



#loss and optimization functions
class Training:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
	
	def __init__(self, num_labels, lr, epochs, print_every, clip, train_loader, valid_loader, path):
		self.lr = lr
		self.epochs = epochs # 3-4 is approx where I noticed the validation loss stop decreasing
		self.print_every = print_every
		self.clip=clip # gradient clipping
		#self.batch_size = batch_size
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.path = path 
		self.num_labels = num_labels

	def save_model(self, model):
		saved_trainer = copy.deepcopy(model)
		with open(self.path + "my_trainer_object.pkl", "wb") as output_file:
		    pickle.dump(saved_trainer, output_file)

	def epoch_loop(self):

				# Instantiate the model w/ hyperparams
		vocab_size = len(vocab_to_int) + 1 # +1 for zero padding + our word tokens
		output_size = 1
		embedding_dim = 512 
		hidden_dim = 768
		n_layers = 2

		net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
		
		#net = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.num_labels)
		print(net)


		criterion = nn.BCELoss()
		optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)

		#obtain one batch of training data
		dataiter = iter(self.train_loader)
		sample_x, sample_y = dataiter.next()

		# First checking if GPU is available
		train_on_gpu=torch.cuda.is_available()

		if(train_on_gpu):
		    print('Training on GPU.')
		else:
		    print('No GPU available, training on CPU.')

		# training params			
		# move model to GPU, if available
		if(train_on_gpu):
		    net.cuda()

		net.train()
		# train for some number of epochs
		counter = 0
		for e in range(epochs):
    # batch loop	
		    for inputs, labels in train_loader:
		        curr_batch_size = inputs.size(0)
		        # initialize hidden state
		        h = net.init_hidden(curr_batch_size)
		        #h = net(curr_batch_size, labels =labels)
		        counter += 1

		    # # batch loop
		    # for inputs, labels in train_loader:
		    #     
		        if(train_on_gpu):
		            inputs, labels = inputs.cuda(), labels.cuda()

		        # Creating new variables for the hidden state, otherwise
		        # we'd backprop through the entire training history
		        h = tuple([each.data for each in h])

		        # zero accumulated gradients
		        net.zero_grad()

		        # get the output from the model
		        output, h = net(inputs, h)

		        # calculate the loss and perform backprop
		        loss = criterion(output.squeeze(), labels.float())
		        loss.backward()
		        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
		        nn.utils.clip_grad_norm_(net.parameters(), self.clip)
		        optimizer.step()

		        # loss stats
		        if counter % self.print_every == 0:
		            # Get validation loss
		            val_h = net.init_hidden(curr_batch_size)
		            #val_h = net(curr_batch_size, labels =labels)
		            val_losses = []
		            net.eval()
		            for inputs, labels in self.valid_loader:

		                # Creating new variables for the hidden state, otherwise
		                # we'd backprop through the entire training history
		                val_h = tuple([each.data for each in val_h])

		                if(train_on_gpu):
		                    inputs, labels = inputs.cuda(), labels.cuda()

		                output, val_h = net(inputs, val_h)
		                val_loss = criterion(output.squeeze(), labels.float())

		                val_losses.append(val_loss.item())

		            net.train()
		            print("Epoch: {}/{}...".format(e+1, self.epochs),
		                  "Step: {}...".format(counter),
		                  "Loss: {:.6f}...".format(loss.item()),
		                  "Val Loss: {:.6f}".format(np.mean(val_losses)))

		self.save_model(net)
		return net

	def testing(self):
		#read model
		with open(self.path + "my_trainer_object.pkl", 'rb') as f:
			net = pickle.load(f)
		# net = BertForSequenceClassification.from_pretrained('bert-base-uncased')


		# Get test data loss and accuracy
		criterion = nn.BCELoss()
		test_losses = [] # track loss
		num_correct = 0

		# init hidden state

		# for inputs, labels in train_loader:
	 #        curr_batch_size = inputs.size(0)
	 #        # initialize hidden state
	 #        h = net.init_hidden(curr_batch_size)

		net.eval()
		# iterate over test data
		for inputs, labels in test_loader:
			#curr_batch_size = inputs.size(0)
		    # Creating new variables for the hidden state, otherwise
		    # we'd backprop through the entire training history
		    curr_batch_size = inputs.size(0)
		    h = net.init_hidden(curr_batch_size)

		    h = tuple([each.data for each in h])

		    train_on_gpu=torch.cuda.is_available()
		    if(train_on_gpu):
		        inputs, labels = inputs.cuda(), labels.cuda()
		    
		    # get predicted outputs
		    output, h = net(inputs, h)
		    
		    # calculate loss
		    test_loss = criterion(output.squeeze(), labels.float())
		    test_losses.append(test_loss.item())
		    
		    # convert output probabilities to predicted class (0 or 1)
		    pred = torch.round(output.squeeze())  # rounds to the nearest integer
		    
		    # compare predictions to true label
		    correct_tensor = pred.eq(labels.float().view_as(pred))
		    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
		    num_correct += np.sum(correct)


		# -- stats! -- ##
		# avg test loss
		print("Test loss: {:.3f}".format(np.mean(test_losses)))

		# accuracy over all test data
		test_acc = num_correct/len(test_loader.dataset)
		print("Test accuracy: {:.3f}".format(test_acc))



#prepare data
pre = Preprocess()
text, labels = pre.save_txt()
vocab_to_int  = pre.mapping_dict()
text_ints, labels = pre.filter_posts(labels)
path = pre.path



num_labels = 3
#training
seq_length = 100
batch_size = 100

pret = PreTraining(text_ints = text_ints, labels = labels, seq_length = seq_length, batch_size= batch_size)
#train_x, train_y, val_x, val_y, test_x, test_y = t.split_train_test()
train_data, valid_data, test_data, train_loader, valid_loader, test_loader = pret.create_tensors()


# Instantiate the model w/ hyperparams
vocab_size = len(vocab_to_int) + 1 # +1 for zero padding + our word tokens
output_size = 1
embedding_dim = 512 
hidden_dim = 768
n_layers = 2

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)


lr=0.001
epochs = 1 # 3-4 is approx where I noticed the validation loss stop decreasing
print_every = 100
clip=5 # gradient clipping

tr = Training(num_labels = num_labels, lr = lr, epochs = epochs, print_every = print_every, clip = clip, train_loader = train_loader, valid_loader = valid_loader, path = path)
train_net = tr.epoch_loop()

# tr.testing()



# #loss and optimization functions
# lr=0.001

# criterion = nn.BCELoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# # check if CUDA is available
# train_on_gpu = torch.cuda.is_available()

# #training params

# epochs = 3 # 3-4 is approx where I noticed the validation loss stop decreasing

# counter = 0
# print_every = 100
# clip=5 # gradient clipping

# #move model to GPU, if available
# if(train_on_gpu):
#     net.cuda()

# net.train()
# # train for some number of epochs
# 		# train for some number of epochs
# counter = 0
# for e in range(epochs):
#     # batch loop
#     for inputs, labels in train_loader:
#         curr_batch_size = inputs.size(0)
#         # initialize hidden state
#         h = net.init_hidden(curr_batch_size)

#         counter += 1

        
#         if(train_on_gpu):
#             inputs, labels = inputs.cuda(), labels.cuda()

#         # Creating new variables for the hidden state, otherwise
#         # we'd backprop through the entire training history
#         h = tuple([each.data for each in h])

#         # zero accumulated gradients
#         net.zero_grad()

#         # get the output from the model
#         output, h = net(inputs, h)

#         # calculate the loss and perform backprop
#         loss = criterion(output.squeeze(), labels.float())
#         loss.backward()
#         # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
#         nn.utils.clip_grad_norm_(net.parameters(), clip)
#         optimizer.step()

#         # loss stats
#         if counter % print_every == 0:
#             # Get validation loss
#             val_h = net.init_hidden(curr_batch_size)
#             val_losses = []
#             net.eval()
#             for inputs, labels in valid_loader:

#                 # Creating new variables for the hidden state, otherwise
#                 # we'd backprop through the entire training history
#                 val_h = tuple([each.data for each in val_h])

#                 if(train_on_gpu):
#                     inputs, labels = inputs.cuda(), labels.cuda()

#                 output, val_h = net(inputs, val_h)
#                 val_loss = criterion(output.squeeze(), labels.float())

#                 val_losses.append(val_loss.item())

#             net.train()
#             print("Epoch: {}/{}...".format(e+1, epochs),
#                   "Step: {}...".format(counter),
#                   "Loss: {:.6f}...".format(loss.item()),
#                   "Val Loss: {:.6f}".format(np.mean(val_losses)))
# # lr=0.001
# epochs = 1 # 3-4 is approx where I noticed the validation loss stop decreasing
# print_every = 100
# clip=5 # gradient clipping

# tr = Training(lr = lr, epochs = epochs, print_every = print_every, clip = clip,  batch_size = batch_size, train_loader = train_loader, valid_loader = valid_loader)
# train_net = tr.epoch_loop()

# tr.testing()













