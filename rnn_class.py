"""
A minimal RNN class written only using Python and Numpy for expositional purposes.
Compare this with the rnn.py code (a refactored version of andrej karpathy's code)

It is highly trimmed down, but has all the right basics (and also great for showing
off the vanishing gradient problem)

It is trained using gradient descent, with gradients being found using standard 
backpropagation with a quadratic cost function and a tanh activation function.
"""

import random
import numpy as np

# Read Data and Print Statistics
data = open("input.txt", "r").read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print "data has {0} characters of which {1} are unique".format(data_size, vocab_size)

print enumerate(chars)
# Make a hashmap (dictionary) to index the characters
char_to_ix = {ch:i for i, ch in enumerate(chars)}
ix_to_char = {i:ch for i, ch in enumerate(chars)}



class SimpleRNN(object):
	"""
	Simple Vanilla RNN class that creates 3 layer RNN, input->hidden->output
	for next character prediction.

	Uses standard backpropagation to compute learning gradients and tuned by gradient
	descent
	"""
	def __init__(self, text_df, hidden_layer_size, sequence_length, learning_rate):
		### Data given to the model ###
		self.data = open(text_df, "r").read()
		self.vocab = list(set(data))
		self.vocab_size = len(vocab)

		### Parameters of the model (ie weights and biases) ###
		self.W_xh = np.random.randn(hidden_layer_size, vocab_size)*0.01 # weights from input into hidden
		self.b_h = np.zeros((hidden_layer_size,1)) # bias of hidden layer
		self.W_hh = np.random.randn((hidden_layer_size, hidden_layer_size)) # weights from hidden to hidden
		# no bias from hidden to hidden loop
		self.W_hy = np.random.randn((vocab_size, hidden_layer_size))*0.01# weights from hidden to output
		self.b_y = np.zeros((vocab_size,1))

		### Hyperparameters ###
		self.hidden_layer_size = hidden_layer_size
		self.sequence_length = sequence_length
		self.learning_rate = learning_rate

		### Internal index of vocab ###
		# Mainly helpful rather than necessary (especially for sampling from our network)
		self.char_to_ix = {ch:i for i, ch in enumerate(vocab)}
		self.ix_to_char = {i:ch for i, ch in enumerate(vocab)}


	def feedforward(self, a):
		"""
		Return the output vector if "a" is given as input 
		"""