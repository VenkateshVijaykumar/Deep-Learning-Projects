import numpy as np
import gym
from gym import wrappers

def val_iter(env, gamma = 1.0):
    v = np.zeros(env.env.nS)
    max_iter = 1000000
    delta = 1e-20
    for i in range(max_iter):
        previous_val = np.copy(v)
        for state in range(env.env.nS):
            Q_sa = [sum([p*(r + previous_val[s_]) for p, s_, r, _ in env.env.P[state][a]]) for a in range(env.env.nA)]
            v[state] = max(Q_sa)
        if (np.sum(np.fabs(previous_val - v)) <= delta):
            print ('value iteration converged at {} iterations' .format(i+1))
            break
    return v

def get_policy(env, v, probs, gamma = 1.0):
    policy = np.zeros(env.env.nS)
    choice_arr = ["q","r"]
    for state in range(env.env.nS):
        Q_sa = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for next_SR in env.env.P[state][action]:
                p,  s_, r, _ = next_SR
                Q_sa[action] += (p * (r + gamma * v[s_]))
        if(np.random.choice(choice_arr,1,p=probs)[0]) == "r":
            policy[state] = Q_sa[env.action_space.sample()]
        else:
            policy[state] = np.argmax(Q_sa)
    return policy

def run_episode(env, policy, gamma, render = False):
    obs = env.reset()
    total_reward = 0
    index = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += (gamma ** index * reward)
        index += 1
        if done:
            break
    return total_reward

def score_policy(env, policy, gamma = 1.0, n = 100):
    scores = [run_episode(env, policy, gamma = gamma, render = False) for _ in range(n)]
    return np.mean(scores)

def main():
    env_name = 'FrozenLake8x8-v0'
    gamma = 1.0
    scores = []
    probs = [0,0]
    for p2 in range(0,11):
        p1 = 10.0-p2
        prob2 = float(p2)/10.0
        prob1 = float(p1)/10.0
        probs[0] = prob1
        probs[1] = prob2
        #print(probs)
        #gamma = float(g)/10.0
        env = gym.make(env_name)
        bestV = val_iter(env, gamma)
        policy = get_policy(env, bestV, probs, gamma)
        pol_score = score_policy(env, policy, gamma, n = 1000)
        scores.append(pol_score)
        np.savetxt('vi_randVSscore8x8.txt', scores)
        print ('Policy mean score: {} at: {} percent randomness' .format(pol_score, prob2*100))

if __name__ == '__main__':
    main()