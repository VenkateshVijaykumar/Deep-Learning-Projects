''' 
This code is written as an attempt at implementing Q learning on a simple domain
The code utilizes the Taxi-v2 environment found at: gym.openai.com
The Taxi problem was introduced in Dietterich(2000). It is a grid-based domain where the goal of
the agent is to pick up a passenger at one location and drop them off in another. There are 4
fixed locations, each assigned a different letter. The agent has 6 actions; 4 for movement, 1 for
pickup, and 1 for dropoff. The domain has a discrete state space and deterministic transitions.
'''

import gym
import numpy as np

env = gym.make("Taxi-v2")
env.reset()
Q = np.zeros([env.observation_space.n, env.action_space.n])
alpha = 1
gamma = 0.9
epsilon = 0.4
file_handle = file('answershw3.txt', 'a')

def epsilon_greedy(state, epsilon):
	if np.random.uniform(0,1) < epsilon:
		return env.action_space.sample()
	else:
		return np.argmax(Q[state])

def update():
	Q[state, action] += (alpha * (reward + (gamma * (np.max(Q[state2]))) - Q[state, action]))

for episode in range(1, 1001):
	done = False
	G, reward = 0, 0
	state = env.reset()
	while done != True:
		action = epsilon_greedy(state, epsilon)
		state2, reward, done, info = env.step(action)
		update()
		G += reward
		state = state2
		#epsilon /= 10
		if done == True:
			Q[state, action] += (alpha * (reward + (0 - Q[state, action])))

answers = []

print Q[462, 4]
print Q[398, 3]
print Q[253, 0]
print Q[377, 1]
print Q[83, 5]

answers.append(Q[42, 4])
answers.append(Q[472, 5])
answers.append(Q[274, 4])
answers.append(Q[292, 0])
answers.append(Q[158, 2])
answers.append(Q[393, 0])
answers.append(Q[71, 5])
answers.append(Q[266, 0])
answers.append(Q[352, 5])
answers.append(Q[281, 3])

np.savetxt(file_handle, answers)
file_handle.close()