import numpy as np
import gym
from gym import wrappers

def compute_policy_v(env, policy, gamma=1.0):
    v = np.zeros(env.env.nS)
    eps = 1e-10
    choice_arr = ["q","r"]
    while True:
        prev_v = np.copy(v)
        for s in range(env.env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break
    return v

def get_policy(env, v, gamma = 1.0):
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        q_sa = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

def run_episode(env, policy, gamma = 1.0, render = False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def score_policy(env, policy, gamma = 1.0, n = 100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

def policy_iteration(env, gamma = 1.0):
    policy = np.random.choice(env.env.nA, size=(env.env.nS))
    max_iterations = 200000
    gamma = 1.0
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = get_policy(env, old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy
    return policy

def main():
    env_name  = 'FrozenLake-v0'
    env = gym.make(env_name)
    scores_arr = []
    gamma = 1.0
    optimal_policy = policy_iteration(env, gamma = gamma)
    scores = score_policy(env, optimal_policy, gamma = gamma)
    print('Average scores: {} at gamma {}' .format((np.mean(scores)),gamma))
    
if __name__ == '__main__':
    main()