import sys
import gym
import pylab
import random
import numpy as np
import h5py
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 100


# DQN Agent for the LunarLander
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQNAgent_test:
    def __init__(self, state_size, action_size):
        # if you want to see LunarLander learning, then change to True
        self.render = False
        self.load_model = True #False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 0.3 #1.0
        self.batch_size = 64
        self.train_start = 1000
        # create replay memory using deque
        self.memory = deque(maxlen=20000000)
        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()
        # initialize target model
        self.update_target_model()
        if self.load_model:
            self.model.load_weights("./save_model/lunarlander5k_dqn.h5")

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model
    def get_action(self, state):
        q_value = self.model.predict(state)
        return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # pick samples randomly from replay memory (with batch_size)
    def update_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)
        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []
        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])
        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)
        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (np.amax(target_val[i]))
        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent_test(state_size, action_size)
    scores, episodes = [], []
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        while not done:
            if agent.render:
                env.render()
            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            if not done or score == 199:
                reward = reward
            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)
            # every time step update
            agent.update_model()
            score += reward
            state = next_state
            if done:
                # every episode update the target model to be same with model
                agent.update_target_model()
                # every episode, plot the play time
                score = score if score == 200 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/lunarlander_dqn_test.png")
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)
        # save the model
        if e % 50 == 0:
            agent.model.save_weights("./save_model/lunarlander_dqn_test.h5")