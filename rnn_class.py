"""
A minimal RNN class written only using Python and Numpy for expositional purposes.
Compare this with the rnn.py code (a refactored )

It is highly trimmed down, but has all the right basics (and also great for showing
off the vanishing gradient problem)

It is trained using gradient descent, with gradients being found using standard 
backpropagation with a quadratic cost function and a tanh activation function.
"""