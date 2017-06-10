import numpy as np
import _pickle as _pickle
import gym

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-1#for convergence (too low- slow to converge, too high,never converge)
gamma = 0.99 # discount factor for reward (i.e later rewards are exponentially less important)
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?

D = 80 * 80
if resume:
  model = pickle.load(open('save.p', 'rb')) #load from pickled checkpoint
else:
  model = {} #initialize model 

  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() }
## rmsprop (gradient descent) memory used to update model
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() }

#activation function
def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x))

#takes a single game frame as input
#preprocesses before feeding into model
#Below function is based on some one's work. 
def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel() #flattens 


def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  #initilize discount reward matrix as empty
  discounted_r = np.zeros_like(r)
  #to store reward sums
  running_add = 0
  #for each reward
  for t in reversed(range(0, r.size)):
    #if reward at index t is nonzero, reset the sum, since this was a game boundary (pong specific!)
    if r[t] != 0: running_add = 0 
    #increment the sum 
    #https://github.com/hunkim/ReinforcementZeroToAll/issues/1
    running_add = running_add * gamma + r[t]
    #earlier rewards given more value over time 
    #assign the calculated sum to our discounted reward matrix
    discounted_r[t] = running_add
  return discounted_r

#forward propagation via numpy woot!
def policy_forward(x):
  #matrix multiply input by the first set of weights to get hidden state
  #will be able to detect various game scenarios (e.g. the ball is in the top, and our paddle is in the middle)
  h = np.dot(model['W1'], x)
  #apply an activation function to it
  #f(x)=max(0,x) take max value, if less than 0, use 0
  h[h<0] = 0 # ReLU nonlinearity
  #repeat process once more
  #will decide if in each case we should be going UP or DOWN.
  logp = np.dot(model['W2'], h)
  #squash it with an activation (this time sigmoid to output probabilities)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  #recursively compute error derivatives for both layers, this is the chain rule
  #epdlopgp modulates the gradient with advantage
  #compute updated derivative with respect to weight 2. It's the parameter hidden states transpose * gradient w/ advantage (then flatten with ravel())
  dW2 = np.dot(eph.T, epdlogp).ravel()
  #Compute derivative hidden. It's the outer product of gradient w/ advatange and weight matrix 2 of 2
  dh = np.outer(epdlogp, model['W2'])
  #apply activation
  dh[eph <= 0] = 0 # backpro prelu
  #compute derivative with respect to weight 1 using hidden states transpose and input observation
  dW1 = np.dot(dh.T, epx)
  #return both derivatives to update weights
  return {'W1':dW1, 'W2':dW2}

