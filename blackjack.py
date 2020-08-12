import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

env=gym.make('Blackjack-v0')

"""Index Method"""

def get_index(state):
  # a: 2 -> 21
  # b: 1 -> 10
  # c: 0 or 1
 
  a, b, c = state
  return ((20*10*c)+(10*(b-1))+(a-2))

"""Loop to train the agent"""

# 400 = number of states = hand_sum(2-21, 20 possibilities) * dealer_card(1-10, 10 possibilities) * useable ace?(0 or 1, 2 possibilities)
table = np.zeros((400, env.action_space.n))

episodes = 100000
alpha = 0.05 # step size
gamma = 0.9 # discount
epsilon = 0.95 # for randomness
decay = 0.999 # randomness decreases over time

rewards = []

for i in range(episodes):

  state = env.reset()
  done = False

  while not done:

    index = get_index(state)
    
    random_num = np.random.random_sample()
    if random_num >= epsilon:
      #take the action with the max value
      action = np.argmax(table[index]) # 0 = stick, 1 = hit
    else:
      #choose a random action
      action = np.random.randint(2) # 0 = stick, 1 = hit

    new_state,reward,done,_=env.step(action)

    # Update rule
    target = reward + gamma * np.max(table[get_index(new_state)])
    table[index, action] = table[index, action] + alpha*(target - table[index,action])

    state = new_state
    rewards.append(reward)

    epsilon *= decay

count = 0
range = 20
averages = []
while count < episodes-range:
  averages.append(np.mean(rewards[count:count+range]))
  count+=range

plt.plot(percentages)
plt.show()