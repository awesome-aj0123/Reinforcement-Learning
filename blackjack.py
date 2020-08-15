import gym
import matplotlib.pyplot as plt
import numpy as np
import random

env=gym.make('Blackjack-v0')

table = np.zeros((32, 11, 2, env.action_space.n))

episodes = 100000
alpha = 0.01 # step size
gamma = 0.9 # discount
epsilon = 0.95 # for randomness
decay = 0.001 # randomness decreases over time

rewards = []

for i in range(episodes):

  state = env.reset()
  done = False

  state = list(state)
  state[2] = int(state[2])
  state[0] -= 2
  state[1] -= 1
  state = tuple(state)

  while not done:

    # choosing action value
    random_num = np.random.random_sample()
    if random_num >= epsilon:
      #take the action with the max value
      action = np.argmax(table[state]) # 0 = stick, 1 = hit
    else:
      #choose a random action
      action = env.action_space.sample() # 0 = stick, 1 = hit

    new_state, reward, done, _ = env.step(action)

    new_state = list(new_state)
    new_state[2] = int(new_state[2])
    new_state[0] -= 2
    new_state[1] -= 1
    new_state = tuple(new_state)

    target = reward + gamma * np.max(table[new_state])
    table[state][action] = table[state][action] + alpha*(target - table[state][action])

    state = new_state

  rewards.append(reward)
  epsilon *= 0.01 + np.exp(-decay*i)

sums = 0
avg_rewards = []
for i in range(len(rewards)):
  if rewards[i] == 1:
    sums+=1;
  avg_rewards.append(sums / (i+1));

plt.ylim(0, 0.5)
plt.plot(avg_rewards)
plt.show()
