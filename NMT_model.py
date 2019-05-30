import pickle, math, time
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed, bidirectional, Dropout, GRU, RepeatVector

# Fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)

	return tokenizer

# Max sentence length
def max_length(lines):

	return max(len(line.split()) for line in lines)

# Encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# Integer encode sequences
	X = tokenizer.texts_to_sequences(lines)

	# Pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')

	return X

# One hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = np.asarray(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)

	return y

# Define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(LSTM(n_units))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))

	return model

def train_model(x, y)
	dataset = x + y
	train = x
	test = y

	# Train test split
	xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)

	# Create tokenizer
	print('\nCreating Topkenizer...')
	tokenizer = create_tokenizer(dataset)

	# Prepare source size and length
	print('\nCreating source size and length')
	src_vocab_size = len(tokenizer.word_index) + 1

	# Prepare target size and length
	print('\nCreating target size and length')
	src_vocab_size = len(tokenizer.word_index) + 1

	length = max_length(dataset)

	# Prepare training data
	print('Preparing/Encoding training data')
	trainX = encode_sequences(tgt_tokenizer, tgt_length, train)
	trainY = encode_sequences(src_tokenizer, src_length, train)
	trainY = encode_output(trainY, src_vocab_size)
	
	# Prepare validation data
	print('Preparing/Encoding validation data')
	testX = encode_sequences(tgt_tokenizer, tgt_length, test)
	testY = encode_sequences(src_tokenizer, src_length, test)
	testY = encode_output(trainY, src_vocab_size)

	# Define model
	model = define_model(tgt_vocab_size, src_vocab_size, tgt_length, src_length, 256)
	modile.compile(optimizer='adam', loss='categorical_crossentropy')
	
	# Summarize defined model (optional)
	print(model.summary())
	
	# Fit model
	print('Fitting model')
	filename = 'en-de_NMT_model.pkl'
	model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(testX, testY), verbose=2)	

	# Save model and word mappings
	model_filename = ''
	model.save(model_filename)
	with open('word_mappings', 'wb') as wm:
		pickle.dump(tokenizer, wm)

	return model_filename