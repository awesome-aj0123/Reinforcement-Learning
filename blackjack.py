import gym
#import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

env=gym.make('Blackjack-v0')

total_table_ace = np.zeros((32, 11))
total_table_no_ace = np.zeros((32, 11))

# 400 = number of states = hand_sum(2-21, 20 possibilities) * dealer_card(1-10, 10 possibilities) * useable ace?(0 or 1, 2 possibilities)
table = np.zeros((32, 11, 2, env.action_space.n))

episodes = 100000
alpha = 0.05 # step size
gamma = 0.9 # discount
epsilon = 0.95 # for randomness
decay = 0.999 # randomness decreases over time

rewards = []

for i in range(episodes):

	state = env.reset() # reset returns (sum_hand(self.player), self.dealer[0], usable_ace(self.player))
	done = False

	state = list(state)
	state[2] = int(state[2])
	state = tuple(state)

	while not done:

		#index = get_index(state)
		
		random_num = np.random.random_sample()
		if random_num >= epsilon:
			#take the action with the max value
			action = np.argmax(table[state]) # 0 = stick, 1 = hit
		else:
			#choose a random action
			action = np.random.randint(2) # 0 = stick, 1 = hit

		new_state, reward, done, _ = env.step(action)

		new_state = list(new_state)
		new_state[2] = int(new_state[2])
		new_state = tuple(new_state)

		if new_state[2]:
			total_table_ace[new_state[0:2]]+=reward
		else:
			total_table_no_ace[new_state[0:2]]+=reward

		# Update rule
		target = reward + gamma * np.max(table[new_state])
		table[state, action] = table[state, action] + alpha*(target - table[state,action])

		state = new_state
		rewards.append(reward)

		epsilon *= decay

fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')

Y = np.arange(0, 32, 1)
X = np.arange(0, 11, 1)
X, Y = np.meshgrid(X, Y)
Z = total_table_ace

ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
					   linewidth=0, antialiased=False)
ax.set_xlabel("Dealer's hand")
ax.set_ylabel("Player's hand")
plt.show()
