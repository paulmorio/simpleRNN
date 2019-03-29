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
	def __init__(self, text_df, hidden_layer_size, sequence_length, learning_rate=0.01):
		### Data given to the model ###
		self.data = open(text_df, "r").read()
		self.vocab = list(set(data))
		self.vocab_size = len(vocab)

		### Parameters of the model (ie weights and biases) ###
		self.W_xh = np.random.randn(hidden_layer_size, self.vocab_size)*0.01 # weights from input into hidden
		self.b_h = np.zeros((hidden_layer_size,1)) # bias of hidden layer
		self.W_hh = np.random.randn((hidden_layer_size, hidden_layer_size)) # weights from hidden to hidden
		# no bias from hidden to hidden loop
		self.W_hy = np.random.randn((self.vocab_size, hidden_layer_size))*0.01# weights from hidden to output
		self.b_y = np.zeros((self.vocab_size,1))

		### Hidden state memory ###
		self.hidden_state_memory = {}
		self.hidden_state_memory[-1] = np.zeros((hidden_size,1)) # initial first memory

		### Hyperparameters ###
		self.hidden_layer_size = hidden_layer_size
		self.sequence_length = sequence_length
		self.learning_rate = learning_rate

		### Internal index of vocab ###
		# Mainly helpful rather than necessary (especially for sampling from our network)
		self.char_to_ix = {ch:i for i, ch in enumerate(vocab)}
		self.ix_to_char = {i:ch for i, ch in enumerate(vocab)}


	def feedforward(self, a, t):
		"""
		Return the output vector if the "a" vector is given as input.

		Params:
		a: array-like denoting the input into the RNN.
			Needs to be the size of the first (input) layer of the Network
		t: time or position in sequence
		"""
		hidden_state = tanh(np.dot(W_xh, a) + np.dot(W_hh, hidden_state_memory[t-1]))
		self.hidden_state_memory[t] = hidden_state # update our memory of the hidden state at time t
		output = np.dot(W_hy, hidden_state)
		return output

	def SGD(self, epochs, test=False):
		"""
		Stochastic gradient descent training RNN for the sequence starting from position pos
		in the data
		"""
		inputs = [self.char_to_ix[ch] for ch in self.data[pos:pos+seq_length]]
		targets = [self.char_to_ix[ch] for ch in self.data[pos+1:pos+seq_length+1]]

	def sample(self, seed_char, n):
		"""
		Sample n letters starting from a seed character from the RNN and print to STDOut
		"""
		sampled_ixs = []
		x = self.make_one_hot_enc(seed_char)

		for t in xrange(n):
			# Predict the next letter
			most_recent_memory_t = max(self.hidden_state_memory.keys())
			h = tanh(np.dot(self.W_xh, x) + np.dot(self.W_hh, self.hidden_state_memory[most_recent_memory_t]))
			y = np.dot(self.W_hy, h) + self.b_y
			y_probs = np.exp(y)/np.sum(np.exp(y)) # Softmax
			predicted_ix = np.random.choice(range(self.vocab_size), p=p.ravel())
			sampled_ixs.append(predicted_ix)

			# next input
			x = np.zeros((self.vocab_size, 1))
			x[predicted_ix] = 1

		sample_text = "".join(self.ix_to_char[ix] for ix in sampled_ixs)
		print "-------------\n %s \n---------------" % (sample_text)


	def make_one_hot_enc(self, char):
		"""
		Returns a one hot encoding of the character given the models vocab
		"""
		x = np.zeros((self.vocab_size, 1))
		seed_index = char_to_ix[seed_char]
		x[seed_index] = 1
		return x







# Static Functions
##################
def tanh(z):
	"""
	Returns the element-wise tanh at z
	"""
	return (np.tanh(z))

def tanh_prime(z):
	"""
	Returns the element-wise derivative of tanh at z
	"""
	return (1.0 - np.tanh(z)**2)
