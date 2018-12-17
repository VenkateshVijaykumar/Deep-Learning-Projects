''' 
This code is written as an attempt at implementing Q learning on a simple domain
The code utilizes the Taxi-v2 environment found at: gym.openai.com
The Taxi problem was introduced in Dietterich(2000). It is a grid-based domain where the goal of
the agent is to pick up a passenger at one location and drop them off in another. There are 4
fixed locations, each assigned a different letter. The agent has 6 actions; 4 for movement, 1 for
pickup, and 1 for dropoff. The domain has a discrete state space and deterministic transitions.
'''

import numpy as np
import gym
from gym import wrappers

def q_learner(gamma,learning_rate,env,episodes):
    Q = np.zeros([env.observation_space.n,env.action_space.n])
    decay_factor = 0.000001
    rewards = []
    max_steps = 100
    for e in range(episodes):
        state = env.reset()
        reward_sum = 0
        done = False
        step = 0
        #learning_rate -= decay_factor
        if learning_rate <= 0:
            break
        for step in range(max_steps):
            step +=1
            a = np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n)*(1.0/(e+1)))
            new_state, reward, done, _ = env.step(a)
            Q[state,a] = Q[state,a] + learning_rate*(reward + gamma*np.max(Q[new_state,:]) - Q[state,a])
            reward_sum += reward
            state = new_state
            if done == True:
                break
        rewards.append(reward_sum)
    return Q, rewards

if __name__ == '__main__':
    env_name  = 'Taxi-v2'
    env = gym.make(env_name)
    episodes = 10000
    gamma = 1.0#0.99
    learning_rate = 0.2
    Q, rewards = q_learner(gamma, learning_rate, env, episodes)
    print('Reward sum on all episodes: ', sum(rewards)/episodes)
    print('Final Q table: \n')
    print(Q)