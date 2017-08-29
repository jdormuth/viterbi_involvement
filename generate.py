from __future__ import print_function

# Load LSTM network and generate text
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Embedding
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import time
import os


file_list = sorted(os.listdir('./weights/'))

# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length))

# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)

pattern = dataX[start]
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

first = ''.join([int_to_char[value] for value in pattern])

# one hot encode the output variable
y = np_utils.to_categorical(dataY)

epoch_list = [1,2,4,8,16,32,40,43]

for j in epoch_list:

	pattern = dataX[start]
	pattern = pattern[:100]

	# define the LSTM model
	model = Sequential()
	model.add(Embedding(256,20, input_length = 100))
	model.add(LSTM(256, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(256))
	model.add(Dropout(0.2))
	model.add(Dense(y.shape[1], activation='softmax'))
	# load the network weights
	filename = "./weights/%s" % file_list[j-1]
	model.load_weights(filename)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	if(j == 1):
		print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
	
		print("Starting phrase: ", end = '') 
		print(first)
	print("Model Generation: Epoch ", j)
	print('\n')
	# generate characters
	pred_list = []
	for i in range(100):
		#time.sleep(.3)
		
		x = numpy.reshape(pattern, (1, len(pattern)))
		#print(pattern)
		prediction = model.predict(x, verbose=0)

		index = numpy.argmax(prediction)
		result = int_to_char[index]
		sys.stdout.write(result)
		sys.stdout.flush()
		time.sleep(.001)
		seq_in = [int_to_char[value] for value in pattern]
		#sys.stdout.write(result)
		pattern.append(index)
		pattern = pattern[1:len(pattern)]
	
		
	j = j*2	
	print("\n\n\n\n")	
