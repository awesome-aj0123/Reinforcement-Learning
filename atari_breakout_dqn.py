import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

class DQN_Agent:
  
  def __init__(self, state_size, action_size):
    self.state_size  = state_size
    self.action_size = action_size
    
    self.memory = deque(maxlen = 1500)

    #defining hyperparameters for agent
    self.learning_rate = 0.05
    self.discount      = 0.9
    self.epsilon       = 1
    self.epsilon_decay = 0.995
    self.epsilon_min   = 0.01
    
    self.model =  self.create_model()

  def create_model(self):
    model = Sequential()
    model.add(Conv2D(32, (2, 2), activation='relu', input_shape=(210, 160, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(self.action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
    return model

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def act(self, state):

    if np.random.random_sample() >= self.epsilon:
      action = np.argmax(self.model.predict(state.reshape(1, 210, 160, 3)))
    else:
      action = env.action_space.sample()
    return action

  def replay(self, batch_size):
    if len(self.memory) < batch_size:
      return

    minibatch = random.sample(self.memory, batch_size)

    states      = np.array([i[0] for i in minibatch])
    actions     = np.array([i[1] for i in minibatch])
    rewards     = np.array([i[2] for i in minibatch])
    next_states = np.array([i[3] for i in minibatch])
    dones       = np.array([i[4] for i in minibatch])

    targets   = rewards + self.discount * (np.amax(self.model.predict_on_batch(next_states), axis=1))
    targets_f = self.model.predict_on_batch(states)

    ind = np.array([i for i in range(batch_size)])
    targets_f[[ind], [actions]] = targets

    self.model.fit(states, targets_f, epochs=1, verbose=0)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

  def load(self, name):
    self.model.load_weights(name)

  def save(self, name):
    self.model.save_weights(name)

env = gym.make("Breakout-v0")

state_size = env.observation_space.shape
state_size # tuple of state representing image

action_size = env.action_space.n
action_size # 4, value of number of possible actions at each state

dir = 'saved_models/atari_breakout/'

if not os.path.exists(dir):
  os.makedirs(dir)

batch_size = 32
episodes = 500
increment = 25

agent = DQN_Agent(state_size, action_size)

for i in range(episodes):

  state = env.reset()
  done = False

  while not done:

    if i % increment == 0:
      env.render()

    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)

    agent.remember(state, action, reward, next_state, done)

    state = next_state

    if done:
      print(f"episode: {i}/{episodes}, score: {reward}, e: {agent.epsilon}")
      break

  if len(agent.memory) > batch_size:
    agent.replay(batch_size)
  if i % increment == 0:
    agent.save(dir + "weights_" + '{:04d}'.format(i) + ".hdf5")

plt.imshow(env.reset())
plt.show()