"""
rnn_class.py
author: paulmorio (Paul Scherer)
date: March 2019

A minimal RNN class written only using Python and Numpy for expositional purposes.
Compare this with the rnn.py code (a refactored version of andrej karpathy's code)

Hint: One of the differences is that we dont use Adagrad to make distinguishing from
adagrad memory and "memory" in the RNN easier.

It is highly trimmed down, but has all the right basics (and also great for showing
off the vanishing gradient problem)

It is trained using gradient descent, with gradients being found using standard 
backpropagation with a cross entropy cost function and a tanh activation function
for each of the neurons.
"""

import random
import numpy as np

# # Read Data and Print Statistics
# data = open("input.txt", "r").read()
# chars = list(set(data))
# data_size, vocab_size = len(data), len(chars)
# print "data has {0} characters of which {1} are unique".format(data_size, vocab_size)

# print enumerate(chars)
# # Make a hashmap (dictionary) to index the characters
# char_to_ix = {ch:i for i, ch in enumerate(chars)}
# ix_to_char = {i:ch for i, ch in enumerate(chars)}


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
        self.vocab = list(set(self.data))
        self.vocab_size = len(self.vocab)

        ### Parameters of the model (ie weights and biases) ###
        self.W_xh = np.random.randn(hidden_layer_size, self.vocab_size)*0.01 # weights from input into hidden
        self.b_h = np.zeros((hidden_layer_size,1)) # bias of hidden layer
        self.W_hh = np.random.randn(hidden_layer_size, hidden_layer_size)*0.01 # weights from hidden to hidden
        # no bias from hidden to hidden loop
        self.W_hy = np.random.randn(self.vocab_size, hidden_layer_size)*0.01# weights from hidden to output
        self.b_y = np.zeros((self.vocab_size,1))

        ### Hidden state memory ###
        self.hprev = np.zeros((hidden_layer_size,1)) # most recent memory


        ### Hyperparameters ###
        self.hidden_layer_size = hidden_layer_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate

        ### Internal index of vocab ###
        # Mainly helpful rather than necessary (especially for sampling from our network)
        self.char_to_ix = {ch:i for i, ch in enumerate(self.vocab)}
        self.ix_to_char = {i:ch for i, ch in enumerate(self.vocab)}


    def feedforward(self, a):
        """
        Return the output vector if the "a" vector is given as input.

        Params:
        a: array-like denoting the input into the RNN.
            Needs to be the size of the first (input) layer of the Network
        """
        avec = self.make_one_hot_enc(a)
        hidden_state = tanh(np.dot(self.W_xh, avec) + np.dot(self.W_hh, self.hprev))
        output = np.dot(self.W_hy, hidden_state)
        return output


    def SGD(self, epochs):
        """
        Stochastic (sequence) gradient descent training RNN for the sequence starting from position pos
        in the data
        """
        pos = 0
        smooth_loss = -np.log(1.0/self.vocab_size)*self.sequence_length # loss at epoch 0

        for epoch in xrange(epochs):
            # Reset if necessary
            if pos+self.sequence_length+1 >= len(self.data) or epoch == 0:
                hprev = np.zeros((self.hidden_layer_size,1)) # reset RNN memory
                pos = 0 # go from start of data

            # inputs = [self.char_to_ix[ch] for ch in self.data[pos:pos+self.sequence_length]]
            inputs = self.data[pos:pos+self.sequence_length]
            targets = [self.char_to_ix[ch] for ch in self.data[pos+1:pos+self.sequence_length+1]]

            nabla_W_xh, nabla_W_hh, nabla_W_hy, nabla_b_h, nabla_b_y, latest_hidden_state, loss = self.backprop(inputs, targets)

            # Update the parameters using the gradients we have computed using backpropagation
            # Here we use the classic gradient descent update rule (though adagrad is wonderful here)
            self.W_xh += -(self.learning_rate/self.sequence_length) * nabla_W_xh 
            self.W_hh += -(self.learning_rate/self.sequence_length) * nabla_W_hh 
            self.W_hy += -(self.learning_rate/self.sequence_length) * nabla_W_hy 
            self.b_h += -(self.learning_rate/self.sequence_length) * nabla_b_h
            self.b_y += -(self.learning_rate/self.sequence_length) * nabla_b_y

            # update the pointer to latest hidden memory state
            self.hprev = latest_hidden_state

            # Print progress 
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            if epoch % 100 == 0: 
                print 'epoch %d, loss: %f' % (epoch, smooth_loss) # print progress

            # Move along sequence
            pos += self.sequence_length

    def backprop(self, inputs, targets):
        """
        The backpropagation algorithm used to find the gradients of the parameters of the model.
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(self.hprev)
        loss = 0

        nabla_W_xh, nabla_W_hh, nabla_W_hy = np.zeros_like(self.W_xh), np.zeros_like(self.W_hh), np.zeros_like(self.W_hy)
        nabla_b_h, nabla_b_y = np.zeros_like(self.b_h), np.zeros_like(self.b_y) 

        # Forward Pass
        ##################
        for t in xrange(len(inputs)):
            # propagate and record all activations for the length of sequence we are looking at
            # print inputs[t]
            xs[t] = self.make_one_hot_enc(inputs[t])
            hs[t] = tanh(np.dot(self.W_xh, xs[t]) + np.dot(self.W_hh, hs[t-1]) + self.b_h)
            ys[t] = np.dot(self.W_hy, hs[t]) + self.b_y
            ps[t] = np.exp(ys[t])/np.sum(np.exp(ys[t]))
            loss += -np.log(ps[t][targets[t],0]) # Cross Entropy Loss

        # Backward Pass
        ##################
        dhnext = np.zeros_like(hs[0])
        for t in reversed(xrange(len(inputs))):
            # backprop of output
            delta_y = np.copy(ps[t])
            delta_y[targets[t]] -= 1
            nabla_W_hy += np.dot(delta_y, hs[t].T)
            nabla_b_y += delta_y

            # backprop into h
            delta_h = np.dot(self.W_hy.T, delta_y) + dhnext
            delta_hraw = tanh_prime(hs[t]) * delta_h
            nabla_W_hh += np.dot(delta_hraw, hs[t-1].T)
            nabla_b_h += delta_hraw
            dhnext = np.dot(self.W_hh.T, delta_hraw)

            # Backprop towards x (only the weights of course)
            nabla_W_xh += np.dot(delta_hraw, xs[t].T)

            # ONLY IF YOU GOT EXPLODING GRADIENTS
            for dparam in [nabla_W_xh, nabla_W_hh, nabla_W_hy, nabla_b_h, nabla_b_y]:
              np.clip(dparam, -5, 5, out=dparam)

        return nabla_W_xh, nabla_W_hh, nabla_W_hy, nabla_b_h, nabla_b_y, hs[len(inputs)-1], loss

    def sample(self, seed_char, n):
        """
        Sample n letters starting from a seed character from the RNN and print to STDOut
        """
        sampled_ixs = []
        x = self.make_one_hot_enc(seed_char)

        for t in xrange(n):
            # Predict the next letter
            h = tanh(np.dot(self.W_xh, x) + np.dot(self.W_hh, self.hprev))
            y = np.dot(self.W_hy, h) + self.b_y
            y_probs = np.exp(y)/np.sum(np.exp(y)) # Softmax
            predicted_ix = np.random.choice(range(self.vocab_size), p=y_probs.ravel())
            sampled_ixs.append(predicted_ix)

            # next input
            x = np.zeros((self.vocab_size, 1))
            x[predicted_ix] = 1

        sample_text = "".join(self.ix_to_char[ix] for ix in sampled_ixs)
        print "----------------\n %s \n---------------" % (sample_text)

    def make_one_hot_enc(self, char):
        """
        Returns a one hot encoding of the character given the models vocab
        """
        x = np.zeros((self.vocab_size, 1))
        seed_index = self.char_to_ix[char]
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
